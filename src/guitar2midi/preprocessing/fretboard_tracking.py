"""Track fretboard corners through a clip from one annotated frame, and warp to a fixed strip.

Why: one set of corners per clip assumes nothing moves, but GAPS videos have camera zoom/pan, players
moving the guitar, and cuts. Without tracking the fretboard slides out of the strip mid-clip.
This is a first version of Stage 1 steps 2-4 (homography, temporal smoothing, warp).

Method (per direction, forward and backward from the annotated frame, inside its shot only):
    1. mask = neck quad of the previous frame, dilated by ~0.5 neck widths (frets, strings, wood)
    2. Shi-Tomasi features in the mask, pyramidal Lucas-Kanade flow to the next frame,
       forward-backward check (< 1 px)
    3. RANSAC homography (3 px); accepted if >= ``min_inliers`` inliers, the new quad is convex and
       its area changes < 25 % per frame; otherwise the last good corners are held ("lost" frame)
    4. after tracking: centered moving average of the corners over ``smooth`` frames
Frames outside the annotated shot (after a hard cut) get no corners (NaN).

Corner order everywhere: (nut/high-E, nut/low-E, fret20/high-E, fret20/low-E).
"""

from __future__ import annotations

import cv2
import numpy as np

from guitar2midi.preprocessing.fretboard_grid import corners_valid

STRIP_W, STRIP_H = 800, 144


def _quad(corners: np.ndarray) -> np.ndarray:
    return np.asarray(corners, np.float32).reshape(4, 2)[[0, 1, 3, 2]]  # boundary order


def neck_mask(shape: tuple[int, int], corners: np.ndarray, dilate_widths: float = 0.5) -> np.ndarray:
    c = np.asarray(corners, np.float64).reshape(4, 2)
    width = 0.5 * (np.linalg.norm(c[0] - c[1]) + np.linalg.norm(c[2] - c[3]))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, _quad(corners).astype(np.int32), 255)
    k = max(int(dilate_widths * width), 1) * 2 + 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def step_homography(prev: np.ndarray, nxt: np.ndarray, mask: np.ndarray,
                    max_features: int = 400) -> tuple[np.ndarray | None, int]:
    """Homography prev -> nxt from masked LK flow. Returns (H or None, inlier count)."""
    p0 = cv2.goodFeaturesToTrack(prev, max_features, 0.01, 5, mask=mask)
    if p0 is None or len(p0) < 8:
        return None, 0
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, p0, None, **lk)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, p1, None, **lk)
    good = (st1[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0 - p0b, axis=2)[:, 0] < 1.0)
    if good.sum() < 8:
        return None, int(good.sum())
    H, inl = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, 3.0)
    return (H, int(inl.sum())) if H is not None else (None, 0)


def _area(corners: np.ndarray) -> float:
    return float(cv2.contourArea(_quad(corners)))


def track_corners(frames: list[np.ndarray], ref_idx: int, ref_corners: np.ndarray,
                  shot: tuple[int, int] | None = None, min_inliers: int = 15,
                  smooth: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corners for every frame.

    Returns (corners (N, 4, 2) float, NaN outside the shot; tracked (N,) bool, False where the
    tracker lost the neck and held the last good corners or outside the shot; inliers (N,) int).
    """
    n = len(frames)
    a, b = shot if shot is not None else (0, n)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    corners = np.full((n, 4, 2), np.nan)
    tracked = np.zeros(n, bool)
    inliers = np.zeros(n, int)
    ref = np.asarray(ref_corners, np.float64).reshape(4, 2)
    corners[ref_idx], tracked[ref_idx] = ref, True
    for step, stop in ((1, b), (-1, a - 1)):
        cur, prev = ref.copy(), ref_idx
        for t in range(ref_idx + step, stop, step):
            H, k = step_homography(gray[prev], gray[t], neck_mask(gray[prev].shape, cur))
            inliers[t] = k
            if H is not None and k >= min_inliers:
                new = cv2.perspectiveTransform(cur.reshape(-1, 1, 2), H).reshape(4, 2)
                ratio = _area(new) / max(_area(cur), 1e-6)
                if corners_valid(new) and 0.8 < ratio < 1.25:
                    cur, tracked[t] = new, True
            corners[t] = cur
            prev = t
    if smooth > 1:
        corners = smooth_corners(corners, smooth)
    return corners, tracked, inliers


def smooth_corners(corners: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average over frames, ignoring NaN frames (shot boundaries are not crossed
    because frames outside the shot are NaN)."""
    out = corners.copy()
    valid = ~np.isnan(corners[:, 0, 0])
    h = window // 2
    for t in np.flatnonzero(valid):
        lo, hi = max(t - h, 0), min(t + h + 1, len(corners))
        win = corners[lo:hi][valid[lo:hi]]
        out[t] = win.mean(0)
    return out


def strip_homography(corners: np.ndarray, width: int = STRIP_W, height: int = STRIP_H,
                     neck_frac: float = 0.5) -> np.ndarray:
    """Homography image -> strip. Nut on the left, fret 20 on the right, high-E on top; the neck fills
    the middle ``neck_frac`` of the strip height, the rest shows the fingers above/below the neck."""
    m = (1 - neck_frac) / 2 * height
    dst = np.float32([[0, m], [0, height - m], [width, m], [width, height - m]])
    return cv2.getPerspectiveTransform(np.asarray(corners, np.float32).reshape(4, 2), dst)


def warp_strip(frame: np.ndarray, corners: np.ndarray, width: int = STRIP_W, height: int = STRIP_H) -> np.ndarray:
    """Grayscale fretboard strip (height, width) uint8; black if corners are NaN."""
    if np.isnan(corners).any():
        return np.zeros((height, width), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.warpPerspective(gray, strip_homography(corners, width, height), (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------- visual check
def tracking_block(frames: list[np.ndarray], corners: np.ndarray, tracked: np.ndarray, strips: np.ndarray,
                   idx: list[int], label: str = "", flag: bool = False, tile_w: int = 200) -> np.ndarray:
    """Frames ``idx`` with the neck quad (green = tracked, red = held) above their strips, plus a label line."""
    tile_h = tile_w * 9 // 16
    top, bottom = [], []
    for t in idx:
        f = frames[t].copy()
        if not np.isnan(corners[t]).any():
            col = (0, 255, 0) if tracked[t] else (0, 0, 255)
            cv2.polylines(f, [_quad(corners[t]).astype(np.int32)], True, col, max(2, f.shape[1] // 300))
        s = min(tile_w / f.shape[1], tile_h / f.shape[0])
        f = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
        tile = np.zeros((tile_h, tile_w, 3), np.uint8)
        tile[(tile_h - f.shape[0]) // 2:(tile_h - f.shape[0]) // 2 + f.shape[0],
             (tile_w - f.shape[1]) // 2:(tile_w - f.shape[1]) // 2 + f.shape[1]] = f
        top.append(tile)
        bottom.append(cv2.cvtColor(cv2.resize(strips[t], (tile_w, tile_w * STRIP_H // STRIP_W)), cv2.COLOR_GRAY2BGR))
    img = np.vstack([np.hstack(top), np.hstack(bottom)])
    lab = np.full((20, img.shape[1], 3), 255, np.uint8)
    cv2.putText(lab, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if flag else (0, 0, 0), 1)
    return np.vstack([img, lab])


def save_sheet(blocks: list[np.ndarray], dst, per_row: int = 3, gap: int = 8) -> None:
    """Grid of ``tracking_block`` images as one JPEG."""
    if not blocks:
        return
    bw = blocks[0].shape[1] + gap
    bh = max(b.shape[0] for b in blocks)
    n_rows = (len(blocks) + per_row - 1) // per_row
    sheet = np.full((n_rows * bh, per_row * bw, 3), 255, np.uint8)
    for i, b in enumerate(blocks):
        rr, cc = divmod(i, per_row)
        sheet[rr * bh:rr * bh + b.shape[0], cc * bw:cc * bw + b.shape[1]] = b
    cv2.imwrite(str(dst), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
