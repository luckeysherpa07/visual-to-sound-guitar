"""Video helpers for the fretboard preprocessing: frame reading and hard-cut (shot) detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frames(path: str | Path) -> list[np.ndarray]:
    """All frames of a (short) video as BGR arrays."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def detect_cuts(frames: list[np.ndarray], hist_threshold: float = 0.6, diff_min: float = 15.0,
                diff_ratio: float = 6.0) -> list[int]:
    """Indices t where a hard cut happens between frame t-1 and t.

    A cut is either a colour change (HSV histogram correlation < ``hist_threshold``) or a jump in mean
    absolute gray difference (64x36 thumbnails) above ``diff_min`` and ``diff_ratio`` x the clip's median
    frame difference (catches cuts between similarly coloured shots; ignores normal motion).
    Gradual transitions (dissolves) are not detected; the tracker's quality check catches those.
    """
    if len(frames) < 2:
        return []
    hists = [_hist(f) for f in frames]
    thumbs = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(float)
              for f in frames]
    diff = np.array([np.abs(thumbs[t] - thumbs[t - 1]).mean() for t in range(1, len(frames))])
    limit = max(diff_min, diff_ratio * float(np.median(diff)))
    return [t for t in range(1, len(frames))
            if cv2.compareHist(hists[t - 1], hists[t], cv2.HISTCMP_CORREL) < hist_threshold
            or diff[t - 1] > limit]


def shots(n_frames: int, cuts: list[int]) -> list[tuple[int, int]]:
    """[start, end) frame ranges between cuts."""
    bounds = [0, *cuts, n_frames]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def longest_shot(n_frames: int, cuts: list[int]) -> tuple[int, int]:
    return max(shots(n_frames, cuts), key=lambda s: s[1] - s[0])
