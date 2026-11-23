"""Note- and frame-level metrics for transcription output (reused for the final model).

All functions take MIDI files or note arrays (onset, offset, pitch) in seconds.

- note onset P/R/F1 (mir_eval, onset tolerance 50 / 100 ms, no offset)  -- main metric, as in V2N and GAPS
- note onset+offset P/R/F1 (mir_eval defaults: offset ratio 0.2, min 50 ms)
- frame P/R/F1 on a 25 fps grid (the video frame rate)
- chroma similarity: mean cosine similarity of 12-bin pitch-class vectors per frame
  (does the output at least get the harmony / key right?)
- onset correlation: Pearson correlation of smoothed onset-count curves
  (is the output's rhythm synced to the real playing, regardless of pitch?)
- timing F1: onset-time F1 ignoring pitch (mir_eval.onset, 50 ms); simultaneous onsets (chords)
  within 30 ms are merged first, so it scores "a pluck happens here", not how many notes
"""

from __future__ import annotations

from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

FPS = 25


def load_notes(midi_file: str | Path) -> np.ndarray:
    """(N, 3) array of onset, offset, MIDI pitch; drums ignored; sorted by onset."""
    midi = pretty_midi.PrettyMIDI(str(midi_file))
    notes = [(n.start, n.end, n.pitch) for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    arr = np.array(notes, dtype=float).reshape(-1, 3)
    return arr[np.argsort(arr[:, 0], kind="stable")]


def _mir(notes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intervals = notes[:, :2].copy()
    intervals[:, 1] = np.maximum(intervals[:, 1], intervals[:, 0] + 1e-3)  # mir_eval needs end > start
    return intervals, mir_eval.util.midi_to_hz(notes[:, 2])


def _prf(tp: float, n_est: int, n_ref: int) -> tuple[float, float, float]:
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def note_prf(ref: np.ndarray, est: np.ndarray, onset_tolerance: float = 0.05,
             with_offset: bool = False) -> tuple[float, float, float]:
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    ri, rp = _mir(ref)
    ei, ep = _mir(est)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=onset_tolerance, offset_ratio=0.2 if with_offset else None)
    return float(p), float(r), float(f)


def merge_onsets(onsets: np.ndarray, gap: float = 0.03) -> np.ndarray:
    """Sorted unique pluck times: onsets closer than ``gap`` to the previous kept one are merged."""
    out = []
    for t in np.sort(onsets):
        if not out or t - out[-1] >= gap:
            out.append(t)
    return np.array(out)


def timing_prf(ref: np.ndarray, est: np.ndarray, window: float = 0.05) -> tuple[float, float, float]:
    """Pitch-agnostic onset P/R/F1 on merged pluck times."""
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    f, p, r = mir_eval.onset.f_measure(merge_onsets(ref[:, 0]), merge_onsets(est[:, 0]), window=window)
    return float(p), float(r), float(f)


def piano_roll(notes: np.ndarray, duration: float, fps: int = FPS) -> np.ndarray:
    """(128, T) binary roll; a note is active in frames [round(on*fps), round(off*fps)), at least 1 frame."""
    n_frames = int(round(duration * fps))
    roll = np.zeros((128, n_frames), bool)
    for on, off, pitch in notes:
        a = int(round(on * fps))
        b = max(int(round(off * fps)), a + 1)
        roll[int(pitch), max(a, 0):min(b, n_frames)] = True
    return roll


def frame_prf(ref: np.ndarray, est: np.ndarray, duration: float, fps: int = FPS) -> tuple[float, float, float]:
    r, e = piano_roll(ref, duration, fps), piano_roll(est, duration, fps)
    return _prf(float((r & e).sum()), int(e.sum()), int(r.sum()))


def chroma_similarity(ref: np.ndarray, est: np.ndarray, duration: float, fps: int = FPS) -> float:
    """Mean cosine similarity of per-frame pitch-class vectors over frames where the reference sounds.

    Frames where the reference is active but the estimate is silent count as 0.
    """
    r = piano_roll(ref, duration, fps)
    e = piano_roll(est, duration, fps)
    rc = np.zeros((12, r.shape[1]))  # fold all 128 pitches by pitch class
    ec = np.zeros((12, e.shape[1]))
    for p in range(128):
        rc[p % 12] += r[p]
        ec[p % 12] += e[p]
    active = rc.sum(0) > 0
    if not active.any():
        return 0.0
    num = (rc * ec).sum(0)[active]
    den = (np.linalg.norm(rc, axis=0) * np.linalg.norm(ec, axis=0))[active]
    return float(np.mean(np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)))


def onset_curve(notes: np.ndarray, duration: float, fps: int = FPS, sigma_frames: float = 1.0) -> np.ndarray:
    """Onset count per frame, smoothed with a Gaussian (sigma in frames)."""
    n_frames = int(round(duration * fps))
    curve = np.zeros(n_frames)
    for on in notes[:, 0] if len(notes) else []:
        i = int(round(on * fps))
        if 0 <= i < n_frames:
            curve[i] += 1
    if sigma_frames > 0:
        x = np.arange(-int(4 * sigma_frames), int(4 * sigma_frames) + 1)
        k = np.exp(-0.5 * (x / sigma_frames) ** 2)
        curve = np.convolve(curve, k / k.sum(), mode="same")
    return curve


def onset_correlation(ref: np.ndarray, est: np.ndarray, duration: float, fps: int = FPS) -> float:
    """Pearson correlation of smoothed onset curves (0 if either is constant, e.g. no notes)."""
    a, b = onset_curve(ref, duration, fps), onset_curve(est, duration, fps)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(ref: np.ndarray, est: np.ndarray, duration: float) -> dict[str, float]:
    """All metrics for one clip."""
    out = {"n_ref": len(ref), "n_est": len(est)}
    for tol in (0.05, 0.1):
        p, r, f = note_prf(ref, est, tol)
        ms = int(tol * 1000)
        out.update({f"onset_p{ms}": p, f"onset_r{ms}": r, f"onset_f{ms}": f})
    out["onset_offset_f50"] = note_prf(ref, est, 0.05, with_offset=True)[2]
    out["timing_f50"] = timing_prf(ref, est, 0.05)[2]
    p, r, f = frame_prf(ref, est, duration)
    out.update(frame_p=p, frame_r=r, frame_f=f)
    out["chroma_sim"] = chroma_similarity(ref, est, duration)
    out["onset_corr"] = onset_correlation(ref, est, duration)
    return out
