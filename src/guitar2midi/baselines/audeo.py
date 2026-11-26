"""Audeo (Su, Liu, Shlizerman, NeurIPS 2020) zero-shot on our fretboard strips (CP14). Runs in env ``guitar2midi``.

Audeo is a piano video -> MIDI model in two stages, both used here with the released weights, no training:

1. Video2RollNet (ResNet-18 variant, ``Video2RollNet.resnet18``): 5 consecutive grayscale frames, each resized to
   900x100 (a tight top view of the full keyboard, low keys left), pixels / 255 -> logits for 51 keys
   (88-key indices 15..65 = MIDI 36..86) for the centre frame. Roll = sigmoid >= 0.4. At the clip edges the
   window is padded as in ``Video2Roll_inference.load_data`` (frame i < 2: [i, i, i, i+1, i+2]; the last two
   frames: [i-2, i-1, i, i, i]).
2. Roll2MidiNet (U-Net generator of a GAN, ``Roll2MidiNet.Generator``): sigmoid of the stage-1 logits for 100
   frames (two 2 s chunks) x 51 keys -> refined roll, >= 0.5 (``Roll2Midi_inference``).
   Deviation: Audeo drops a trailing unpaired chunk and zero-pads short chunks with logit 0 (= probability 0.5);
   we pad the logits with -20 (probability ~0) to a multiple of 100 frames and cut the output back.

Roll -> notes as in Audeo's ``Midi_synth``: every run of consecutive active frames of a key is one note,
onset = first frame / 25 fps, offset = (last frame + 1) / 25 fps. Velocity is not predicted (fixed 64).

Our strips (800x144, nut left, pitch rising to the right) are resized to 900x100 like Audeo's keyboard crops;
the strip direction matches the keyboard (low notes left).

Code: https://github.com/shlizee/Audeo (commit 031dbd7), cloned to ``$AUDEO_REPO`` (default ~/external/Audeo).
Weights: ``Audeo_dataset.zip`` from the link in the repo's issue #4 (the README's Drive folder is gone, 404),
extracted to ``$AUDEO_DATA`` (default ~/external/Audeo_weights/Audeo_dataset):
``Video2Roll_models/Video2RollNet.pth``, ``Roll2Midi_models/Roll2Midi_model/checkpoint-11.tar``
(best epoch per its hyperparams.json). Both load with ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

AUDEO_REPO = Path(os.environ.get("AUDEO_REPO", Path.home() / "external" / "Audeo"))
AUDEO_DATA = Path(os.environ.get("AUDEO_DATA", Path.home() / "external" / "Audeo_weights" / "Audeo_dataset"))
V2R_CKPT = AUDEO_DATA / "Video2Roll_models" / "Video2RollNet.pth"
R2M_CKPT = AUDEO_DATA / "Roll2Midi_models" / "Roll2Midi_model" / "checkpoint-11.tar"

FPS = 25.0
MIN_KEY, MAX_KEY = 15, 65  # 88-key indices used by Audeo
N_KEYS = MAX_KEY - MIN_KEY + 1  # 51
MIN_MIDI = 21 + MIN_KEY  # 36 (88-key index 0 = A0 = MIDI 21)
ROLL_THRESHOLD = 0.4  # Video2Roll_inference
MIDI_THRESHOLD = 0.5  # Roll2Midi_inference
CHUNK = 50  # frames per Audeo chunk (2 s)
IMG_W, IMG_H = 900, 100


def _import_audeo():
    if str(AUDEO_REPO) not in sys.path:
        sys.path.insert(0, str(AUDEO_REPO))
    import Roll2MidiNet
    import Video2RollNet

    return Video2RollNet, Roll2MidiNet


def preprocess(frames: np.ndarray) -> np.ndarray:
    """(T, H, W) uint8 grayscale -> (T, 100, 900) float32 in [0, 1], resized with PIL as in Audeo's transform.

    NEAREST, not today's PIL default (bicubic): Audeo's ``x.resize((900,100))`` ran on Pillow < 7.0, whose default
    was nearest. Checked in 32a_check_audeo: nearest reproduces Audeo's saved logits (max diff 0.05), bicubic not (1.5).
    """
    from PIL import Image

    return np.stack([np.asarray(Image.fromarray(f).resize((IMG_W, IMG_H), Image.NEAREST), dtype=np.float32) / 255.0
                     for f in frames])


def window_indices(n: int) -> list[list[int]]:
    """Audeo's 5-frame window for every frame, with its edge rule (Video2Roll_inference.load_data)."""
    out = []
    for i in range(n):
        if 2 <= i < n - 2:
            out.append([i - 2, i - 1, i, i + 1, i + 2])
        elif i < 2:
            out.append([i, i, i, min(i + 1, n - 1), min(i + 2, n - 1)])
        else:
            out.append([max(i - 2, 0), max(i - 1, 0), i, i, i])
    return out


class AudeoTranscriber:
    def __init__(self, v2r_ckpt: Path = V2R_CKPT, r2m_ckpt: Path = R2M_CKPT, device: str | None = None,
                 batch_size: int = 32):
        import torch

        V2R, R2M = _import_audeo()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.v2r = V2R.resnet18()
        self.v2r.load_state_dict(torch.load(v2r_ckpt, map_location="cpu", weights_only=True))
        self.v2r.to(self.device).eval()
        self.r2m = R2M.Generator((1, N_KEYS, 2 * CHUNK))
        self.r2m.load_state_dict(torch.load(r2m_ckpt, map_location="cpu", weights_only=True)["state_dict_G"])
        self.r2m.to(self.device).eval()

    def roll_logits(self, frames: np.ndarray, preprocessed: bool = False) -> np.ndarray:
        """Stage 1: (T, H, W) uint8 frames -> (T, 51) Video2Roll logits."""
        torch = self.torch
        x = frames if preprocessed else preprocess(frames)
        wins = window_indices(len(x))
        out = []
        with torch.no_grad():
            for b in range(0, len(wins), self.batch_size):
                batch = np.stack([x[w] for w in wins[b:b + self.batch_size]])  # (B, 5, 100, 900)
                out.append(self.v2r(torch.from_numpy(batch).to(self.device)).float().cpu().numpy())
        return np.concatenate(out)

    def refine(self, logits: np.ndarray) -> np.ndarray:
        """Stage 2: (T, 51) logits -> (T, 51) binary roll from Roll2MidiNet (100-frame pieces)."""
        torch = self.torch
        T = len(logits)
        n = -(-T // (2 * CHUNK)) * 2 * CHUNK
        padded = np.full((n, N_KEYS), -20.0, dtype=np.float32)
        padded[:T] = logits
        out = []
        with torch.no_grad():
            for s in range(0, n, 2 * CHUNK):
                roll = torch.sigmoid(torch.from_numpy(padded[s:s + 2 * CHUNK].T.copy()))[None, None].to(self.device)
                gen = self.r2m(roll)  # (1, 1, 51, 100)
                out.append((gen >= MIDI_THRESHOLD).squeeze().T.cpu().numpy())
        return np.concatenate(out)[:T].astype(np.uint8)

    def transcribe(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        logits = self.roll_logits(frames)
        return {"logits": logits, "roll": (1 / (1 + np.exp(-logits)) >= ROLL_THRESHOLD).astype(np.uint8),
                "midi": self.refine(logits)}


def roll_to_notes(roll: np.ndarray, fps: float = FPS) -> list[tuple[float, float, int, int]]:
    """(T, 51) binary roll -> [(onset s, offset s, MIDI pitch, velocity 64)]; one note per run of active frames."""
    notes = []
    padded = np.vstack([np.zeros((1, roll.shape[1]), roll.dtype), roll, np.zeros((1, roll.shape[1]), roll.dtype)])
    d = np.diff(padded.astype(np.int8), axis=0)
    for k in range(roll.shape[1]):
        ons, offs = np.flatnonzero(d[:, k] == 1), np.flatnonzero(d[:, k] == -1)
        notes += [(on / fps, off / fps, MIN_MIDI + k, 64) for on, off in zip(ons, offs)]
    return sorted(notes)


def write_midi(notes, dst: Path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for on, off, pitch, vel in notes:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=on, end=off))
    pm.instruments.append(inst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(dst))


def plot_roll(strips: np.ndarray, out: dict[str, np.ndarray], notes, ref_midi: Path, valid: np.ndarray | None,
              dst: Path, title: str = "", ref_label: str = "ground truth (GAPS)") -> None:
    """Strip thumbnails, reference piano roll, Audeo stage-1 probabilities with the final (Roll2Midi) notes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pretty_midi
    from matplotlib.colors import LogNorm

    T = len(strips)
    dur = T / FPS
    ref = [(n.start, n.end, n.pitch) for i in pretty_midi.PrettyMIDI(str(ref_midi)).instruments for n in i.notes]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 2, 2]}, sharex=True)
    idx = np.linspace(0, T - 1, 6).astype(int)
    axes[0].imshow(np.concatenate([strips[i] for i in idx], axis=1), cmap="gray", aspect="auto", extent=[0, dur, 0, 1])
    axes[0].set_yticks([])
    axes[0].set_title("input strips at 6 evenly spaced times", fontsize=9)
    lo, hi = MIN_MIDI, MIN_MIDI + N_KEYS - 1
    for ax, label in ((axes[1], ref_label),
                      (axes[2], "Audeo zero-shot: stage-1 key probability (roll threshold 0.4) + final notes (blue)")):
        ax.axhspan(40, 83, color="tab:green", alpha=0.07, lw=0)
        ax.set_ylim(lo - 1, hi + 1)
        ax.set_ylabel("MIDI pitch")
        ax.set_title(label, fontsize=9)
        if valid is not None:
            for t in np.flatnonzero(~valid.astype(bool)):
                ax.axvspan(t / FPS, (t + 1) / FPS, color="0.85", lw=0)
    for on, off, p in ref:
        axes[1].plot([on, off], [p, p], color="k", lw=3, solid_capstyle="butt")
    prob = 1 / (1 + np.exp(-out["logits"].astype(float)))
    im = axes[2].imshow(np.clip(prob.T, 1e-4, 1), origin="lower", aspect="auto", cmap="magma_r", norm=LogNorm(1e-4, 1),
                        extent=[0, dur, lo - 0.5, hi + 0.5], alpha=0.8)
    fig.colorbar(im, ax=axes[2], pad=0.01, label="key probability (log)")
    for on, off, p, _ in notes:
        axes[2].plot([on, off], [p, p], color="tab:blue", lw=2, solid_capstyle="butt")
    for on, _, p in ref:
        axes[2].plot(on, p, "x", color="tab:green", ms=4)
    axes[2].set_xlabel("time (s); green band = guitar range E2-B5, green x = reference onsets, grey = masked frames")
    fig.suptitle(title)
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=110)
    plt.close(fig)
