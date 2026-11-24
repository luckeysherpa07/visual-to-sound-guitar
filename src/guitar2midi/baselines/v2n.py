"""V2N zero-shot on guitar fretboard strips (Kim et al., ISMIR 2026). Runs in the ``v2n`` env (envs/v2n.yml).

V2N is not an installable package, so its repo (pinned commit, see envs/v2n.yml) is put on ``sys.path``
and its own model, stitching and decoder code is used unchanged. Only the data loading is ours: V2N's
dataset reads JPEG frames from HDF5; here the input is our (T, 144, 800) uint8 strip array. Test-time
behaviour copied from V2N's ``predict.py`` + ``configs/oofv_1s``:
  - pixels / 255 -> float32 in [0, 1], grayscale (T, H, W), no normalisation, fp32 (no autocast)
  - 1 s windows (25 frames) with stride round(0.5 s * 25) = 13 frames, plus a last window aligned to the end
  - per-window outputs stitched with ``stitch_center``; ``NoteDecoder`` (onset/frame threshold 0.5,
    offset-guided); pitch index + 21 (A0) = MIDI pitch
The strips are 800x144 like V2N's keyboard crops; nut on the left (low pitches on the left, as on the piano).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

V2N_REPO = Path(os.environ.get("V2N_REPO", Path.home() / "external" / "V2N"))
V2N_CKPT = Path.home() / ".cache" / "v2n" / "v2n_pianovam.safetensors"
FPS = 25.0
MIN_MIDI = 21
WINDOW = 25
STRIDE = 13  # V2N: floor(0.5 s * 25 fps + 0.5)


def _import_v2n():
    if str(V2N_REPO) not in sys.path:
        sys.path.insert(0, str(V2N_REPO))
    import predict as v2n_predict  # V2N's predict.py
    from model.decoding import NoteDecoder
    return v2n_predict, NoteDecoder


def window_starts(n_frames: int, window: int = WINDOW, stride: int = STRIDE) -> list[int]:
    """V2N's VPTDataset._iter_segment_starts for a whole clip with include_last_segment=True."""
    last = n_frames - window
    if last < 0:
        raise ValueError(f"clip shorter than one window ({n_frames} < {window} frames)")
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


FLIPS = ("none", "h", "v", "hv")  # h: nut on the right; v: low E on top; hv: both (= rotate 180)
LAYOUTS = ("mid", "top")
PIANO_MEAN, PIANO_STD = 155.0, 85.0  # median per-frame stats of V2N's PianoVAM strips (30g_strip_stats.py, CP10)


def transform_strips(strips: np.ndarray, flip: str = "none", layout: str = "mid", norm: bool = False) -> np.ndarray:
    """Variants of our (T, 144, 800) strips for the CP10 re-check of the zero-shot result.

    Our strips (``fretboard_tracking.strip_homography``) have the neck in rows 36-108 (middle 50 %), nut left,
    high E on top. V2N's piano crops have the keyboard at the top and a 100 px margin below it (hands).
    layout "top": rows 36-144 (neck + the margin below it) stretched to 144 rows -> neck in the top 2/3 and a
    margin below, like the piano crop. Applied after the flip: without "v" the margin kept below the neck is the
    low-E side of the original strip, with "v" it is the high-E side (where the fretting fingers come from).
    norm: rescale each frame to the mean/std of the piano strips (ours are darker and flatter: mean ~100, std ~58).
    """
    s = strips
    if "h" in flip:
        s = s[:, :, ::-1]
    if "v" in flip:
        s = s[:, ::-1, :]
    if layout == "top":
        import cv2

        h, w = s.shape[1:]
        top = h // 4  # 36
        s = np.stack([cv2.resize(np.ascontiguousarray(f[top:]), (w, h), interpolation=cv2.INTER_LINEAR) for f in s])
    elif layout != "mid":
        raise ValueError(layout)
    if norm:
        f = s.astype(np.float32)
        mu = f.mean(axis=(1, 2), keepdims=True)
        sd = f.std(axis=(1, 2), keepdims=True)
        s = np.clip((f - mu) / np.maximum(sd, 1.0) * PIANO_STD + PIANO_MEAN, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(s)


class V2NTranscriber:
    def __init__(self, checkpoint: Path = V2N_CKPT, device: str | None = None, batch_size: int = 2):
        import torch
        from omegaconf import OmegaConf

        v2n_predict, NoteDecoder = _import_v2n()
        self._v2n = v2n_predict
        model_cfg = OmegaConf.load(V2N_REPO / "configs" / "oofv_1s" / "model" / "default.yaml")
        eval_cfg = OmegaConf.load(V2N_REPO / "configs" / "oofv_1s" / "eval" / "default.yaml")
        cfg = OmegaConf.create({"model": model_cfg})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = v2n_predict.instantiate_transcriber(cfg)
        self.model.load_state_dict(v2n_predict.load_transcriber_state_dict(str(checkpoint)))
        self.model.to(self.device).eval()
        dec = eval_cfg.decoding
        self.decoder = NoteDecoder(onset_threshold=dec.onset_threshold, frame_threshold=dec.frame_threshold,
                                   fps=FPS, prefer_onset_predictions=dec.prefer_onset_predictions,
                                   prefer_offset_predictions=dec.prefer_offset_predictions)
        self.batch_size = batch_size
        self.checkpoint = Path(checkpoint)

    def probs(self, strips: np.ndarray) -> dict[str, np.ndarray]:
        """(T, 144, 800) uint8 -> stitched per-frame outputs, each (T, 88): onset/offset/frame probs, velocity."""
        import torch

        x = torch.from_numpy(np.ascontiguousarray(strips))
        starts = window_starts(len(x))
        windows = []
        with torch.no_grad():
            for i in range(0, len(starts), self.batch_size):
                chunk = starts[i:i + self.batch_size]
                batch = torch.stack([x[s:s + WINDOW] for s in chunk]).float().div(255.0).to(self.device)
                out = self.model(pixel_values=batch)
                for b, s in enumerate(chunk):
                    windows.append((s, s + WINDOW, {k: v[b].float().cpu() for k, v in out.items()}))
        merged, start, end = self._v2n.stitch_song_windows(windows, self._v2n.stitch_center)
        assert (start, end) == (0, len(x))
        return {k: v.numpy() for k, v in merged.items()}

    def decode(self, probs: dict[str, np.ndarray]):
        """Stitched outputs -> list of (onset_s, offset_s, midi_pitch, velocity 0-127)."""
        import torch

        notes = self.decoder({k: torch.from_numpy(v) for k, v in probs.items()})
        if len(notes) == 0:
            return []
        iv = notes.intervals.astype(float) / FPS
        vel = np.clip(np.round(notes.velocities.astype(float) * 127), 1, 127).astype(int)
        return [(float(a), float(b), int(p) + MIN_MIDI, int(v))
                for (a, b), p, v in zip(iv, notes.pitches, vel)]


def write_midi(notes, dst: Path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for on, off, pitch, vel in notes:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=on, end=max(off, on + 1 / FPS)))
    pm.instruments.append(inst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(dst))


def plot_roll(strips: np.ndarray, probs: dict[str, np.ndarray], pred_notes, gt_midi: Path, valid: np.ndarray | None,
              dst: Path, title: str = "", ref_label: str = "ground truth (GAPS)") -> None:
    """Strip thumbnails, GT piano roll and V2N onset probabilities with decoded notes, on one time axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pretty_midi

    T = len(strips)
    dur = T / FPS
    gt = [(n.start, n.end, n.pitch) for i in pretty_midi.PrettyMIDI(str(gt_midi)).instruments for n in i.notes]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 2, 2]}, sharex=True)

    idx = np.linspace(0, T - 1, 6).astype(int)
    axes[0].imshow(np.concatenate([strips[i] for i in idx], axis=1), cmap="gray", aspect="auto",
                   extent=[0, dur, 0, 1])
    axes[0].set_yticks([])
    axes[0].set_title("strips at 6 evenly spaced times (nut left, high E on top)", fontsize=9)

    lo, hi = MIN_MIDI, MIN_MIDI + 87
    for ax, label in ((axes[1], ref_label), (axes[2], "V2N zero-shot: onset prob + decoded notes")):
        ax.axhspan(40, 83, color="tab:green", alpha=0.07, lw=0)  # E2-B5, classical guitar range
        ax.set_ylim(lo - 1, hi + 1)
        ax.set_ylabel("MIDI pitch")
        ax.set_title(label, fontsize=9)
        if valid is not None:
            for t in np.flatnonzero(~valid.astype(bool)):
                ax.axvspan(t / FPS, (t + 1) / FPS, color="0.85", lw=0)
    for on, off, p in gt:
        axes[1].plot([on, off], [p, p], color="k", lw=3, solid_capstyle="butt")
    from matplotlib.colors import LogNorm
    im = axes[2].imshow(np.clip(probs["onset_probs"].T.astype(float), 1e-5, 1), origin="lower", aspect="auto",
                        cmap="magma_r", norm=LogNorm(1e-5, 1), extent=[0, dur, lo - 0.5, hi + 0.5], alpha=0.8)
    fig.colorbar(im, ax=axes[2], pad=0.01, label="onset prob (log; decoding threshold 0.5)")
    for on, off, p, _ in pred_notes:
        axes[2].plot([on, off], [p, p], color="tab:blue", lw=2, solid_capstyle="butt")
    for on, _, p in gt:
        axes[2].plot(on, p, "x", color="tab:green", ms=4)
    axes[2].set_xlabel("time (s); green band = guitar range E2-B5, green x = GT onsets, grey = masked frames")
    fig.suptitle(title)
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=110)
    plt.close(fig)
