"""GuiFiR-Vid evaluation clips for the baseline study (CP7).

GuiFiR-Vid test clips are 10 s YouTube excerpts with audio inside the mp4. Labels (test split, manual):
    frame_tab    fret per string at frames where a string is picked (20 / -1 = string not played)
    frame_capo   capo fret
    frame_gt_pose fretboard corners lt/lb/rt/rb (nut + fret 20) in 640x360 coordinates
Annotated frames are 0-based native-fps frame indices (``<clip>_NNN.png``).

There is no MIDI. The reference for scoring is the GAPS model's transcription of the clip's real audio
(pseudo-GT). The tab is used to check that reference ("tab check"): at each tab frame, is the tab pitch
in the transcription? String order and capo handling are not documented, so ``TAB_VARIANTS`` lists the
readings that are tested; the best-fitting one is reported.

Per clip (same layout as the GAPS clips, see ``baselines.clips``):
    video/<clip>.mp4   silent, 25 fps, width <= 1280
    audio/<clip>.wav   the clip's own audio (no sync correction needed: same file as the video)
    tab/<clip>.json    tab events inside the window (time in clip, frets, capo, GT corners in clip pixels)
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

from guitar2midi.data import guifir

NOT_PLAYED = (20, -1)
OPEN_LOW_FIRST = (40, 45, 50, 55, 59, 64)  # E2 A2 D3 G3 B3 E4, standard tuning
# (name, string order, capo added to the fret?)
TAB_VARIANTS = {
    "low_first+capo": (OPEN_LOW_FIRST, True),
    "high_first+capo": (OPEN_LOW_FIRST[::-1], True),
    "low_first": (OPEN_LOW_FIRST, False),
    "high_first": (OPEN_LOW_FIRST[::-1], False),
}


def song_of(clip_id: str) -> str:
    return clip_id.rsplit("_", 1)[0]


def records_by_clip(split: str = "test") -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in guifir.load_split(split):
        out[guifir.clip_id(r)].append(r)
    for recs in out.values():
        recs.sort(key=guifir.frame_index)
    return dict(out)


def tab_pitches(frame_tab: list[int], capo: int, variant: str = "low_first+capo") -> list[int]:
    open_pitches, add_capo = TAB_VARIANTS[variant]
    return [open_pitches[s] + fret + (capo if add_capo else 0)
            for s, fret in enumerate(frame_tab) if fret not in NOT_PLAYED]


def choose_window(times: np.ndarray, duration: float, length: float = 8.0, step: float = 0.25) -> float:
    """Window start with the most tab events; ties -> the most central start."""
    hi = max(duration - length, 0.0)
    starts = np.arange(0.0, hi + 1e-9, step)
    counts = np.array([((times >= s) & (times < s + length)).sum() for s in starts])
    best = np.flatnonzero(counts == counts.max())
    return float(starts[best[np.argmin(np.abs(starts[best] - hi / 2))]])


def cut_audio(src: Path, dst: Path, start: float, length: float) -> None:
    """Audio track of a video, [start, start+length), 16-bit PCM wav at the source rate/channels."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
           "-vn", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def video_props(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = {"fps": cap.get(cv2.CAP_PROP_FPS), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    out["duration"] = out["n_frames"] / out["fps"] if out["fps"] else 0.0
    return out


# ---------------------------------------------------------------- tab check
def tab_recall(midi: pretty_midi.PrettyMIDI, events: list[dict], variant: str,
               onset_tol: float = 0.1, sounding_tol: float = 0.05) -> dict:
    """Share of tab pitches found in ``midi``.

    onset:    a note of that pitch starts within +-onset_tol of the tab frame (tab frames mark picking moments)
    sounding: a note of that pitch is active at the tab frame (+-sounding_tol)
    chroma:   same as onset but pitch class only (octave errors / tuning reading forgiven)
    """
    notes = [n for inst in midi.instruments for n in inst.notes]
    st = np.array([n.start for n in notes]) if notes else np.zeros(0)
    en = np.array([n.end for n in notes]) if notes else np.zeros(0)
    pc = np.array([n.pitch for n in notes]) if notes else np.zeros(0, int)
    hit = {"onset": 0, "sounding": 0, "chroma": 0}
    total = 0
    for ev in events:
        for p in tab_pitches(ev["frame_tab"], ev["capo"], variant):
            total += 1
            near = np.abs(st - ev["t"]) <= onset_tol
            hit["onset"] += bool((near & (pc == p)).any())
            hit["chroma"] += bool((near & (pc % 12 == p % 12)).any())
            hit["sounding"] += bool(((st - sounding_tol <= ev["t"]) & (en + sounding_tol >= ev["t"]) & (pc == p)).any())
    return {"n_tab_pitches": total, **{k: v for k, v in hit.items()}}
