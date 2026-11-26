"""Foley Music (Gan et al., ECCV 2020) inference on our clips (CP15). Runs in env ``foleymusic``.

Model: retrained by us with the authors' code (``~/external/Foley-Music``, commit dcc2638, unmodified; ``torchpie``
stand-in + torch-2 patch in ``foley_music_shim/``) on their released MUSIC acoustic-guitar keypoints + MIDI
(``outputs/runs/2026-11-26_cp15_foleymusic_acoustic_guitar/exp``: ``config.conf`` + ``model_best.pth.tar`` =
lowest validation loss). It never saw our data.

Input: our keypoints (``foley_music_pose``, (T, 3, 65, 1) at 22 fps, Foley Music's scale). The model takes 6 s
(132 frames) and generates up to 512 MIDI events with beam search (beam 5, as in ``test_AtinPiano_MUSIC.py``).
Clips are covered by two 6 s windows, [0, 6) and [L - 6, L); notes with onset < 6 s come from the first, the rest
from the second (L = 8 s for most clips). Clips shorter than 6 s are padded with their last frame; notes after the
clip end are dropped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("FOLEY_MUSIC_REPO", Path.home() / "external" / "Foley-Music"))
SHIM = Path(__file__).resolve().parent / "foley_music_shim"
EXP = Path("outputs/runs/2026-11-26_cp15_foleymusic_acoustic_guitar/exp")
FPS = 22.0


def _setup_path():
    for p in (str(REPO), str(SHIM)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import foley_torch2_patch  # noqa: F401  (torch-1.x nn.Transformer.forward)


class FoleyMusic:
    def __init__(self, exp: Path = EXP, checkpoint: str = "model_best.pth.tar", device: str = "cuda", beam: int = 5):
        _setup_path()
        import torch
        from pyhocon import ConfigFactory

        from core.dataloaders.youtube_dataset import YoutubeDataset
        from core.models import ModelFactory

        self.torch = torch
        self.cfg = ConfigFactory.parse_file(str(exp / "config.conf"))
        self.window = int(self.cfg.get_float("dataset.duration") * self.cfg.get_float("dataset.fps"))  # 132
        self.duration = self.cfg.get_float("dataset.duration")
        self.device = torch.device(device)
        self.model = ModelFactory(self.cfg).build(device=self.device)
        cp = torch.load(exp / checkpoint, map_location="cpu")
        self.model.load_state_dict(cp["state_dict"])
        self.model.eval()
        self.beam = beam
        self.idx = dict(pad_idx=YoutubeDataset.PAD_IDX, sos_idx=YoutubeDataset.SOS_IDX, eos_idx=YoutubeDataset.EOS_IDX)
        self.max_events = 512

    def window_notes(self, pose: np.ndarray) -> list:
        """(132, 3, 65, 1) -> pretty_midi notes of the generated window (times from the window start)."""
        from core import utils

        torch = self.torch
        x = torch.from_numpy(np.transpose(pose.astype(np.float32), (1, 0, 2, 3)))[None].to(self.device)  # (1,3,T,V,M)
        with torch.no_grad():
            events = self.model.generate(x, target_seq_length=self.max_events, beam=self.beam, **self.idx)
        if events.shape[1] <= 0:
            return []
        pm = utils.midi.tensor_to_pm(events.squeeze(), instrument="Acoustic Guitar (nylon)")
        return [n for inst in pm.instruments for n in inst.notes]

    def transcribe(self, pose: np.ndarray, length: float) -> list[tuple[float, float, int, int]]:
        """Clip keypoints (T, 3, 65, 1) at 22 fps, clip length in s -> [(onset, offset, pitch, velocity)]."""
        T, W = len(pose), self.window
        if T < W:
            pose = np.concatenate([pose, np.repeat(pose[-1:], W - T, axis=0)])
            T = W
        starts = [0] if T == W else [0, T - W]
        notes = []
        for k, s in enumerate(starts):
            t0 = s / FPS
            for n in self.window_notes(pose[s:s + W]):
                on = n.start + t0
                if (k == 0 and len(starts) > 1 and on >= self.duration) or (k == 1 and on < self.duration):
                    continue
                if on >= length:
                    continue
                notes.append((on, min(n.end + t0, length), int(n.pitch), int(n.velocity)))
        return sorted(notes)


def write_midi(notes, dst: Path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Guitar (nylon)"))
    for on, off, pitch, vel in notes:
        inst.notes.append(pretty_midi.Note(velocity=max(vel, 1), pitch=pitch, start=on, end=max(off, on + 0.04)))
    pm.instruments.append(inst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(dst))
