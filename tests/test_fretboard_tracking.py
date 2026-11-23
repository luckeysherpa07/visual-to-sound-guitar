"""Tracking on a synthetic textured 'neck' moved by known zoom/pan/rotation."""

import cv2
import numpy as np
import pytest

from guitar2midi.preprocessing import fretboard_tracking as T
from guitar2midi.preprocessing.video import detect_cuts, longest_shot

W, H = 640, 360
CORNERS = np.array([[500, 160], [500, 200], [120, 150], [120, 215]], float)  # nut right, f20 left


def scene(rng):
    img = np.full((H, W, 3), 40, np.uint8)
    cv2.fillConvexPoly(img, T._quad(CORNERS).astype(np.int32), (90, 120, 160))
    for _ in range(600):  # texture: "frets/strings/wood grain"
        x, y = rng.integers(100, 520), rng.integers(140, 225)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 4)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def motion(t):
    """Similarity transform at frame t: slow zoom, pan and rotation."""
    s, ang, tx, ty = 1 + 0.004 * t, 0.15 * t, 1.5 * t, -0.8 * t
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, s)
    M[:, 2] += (tx, ty)
    return np.vstack([M, [0, 0, 1]])


def make_clip(n=40, seed=0):
    base = scene(np.random.default_rng(seed))
    frames, truth = [], []
    for t in range(n):
        Ht = motion(t)
        frames.append(cv2.warpPerspective(base, Ht, (W, H)))
        truth.append(cv2.perspectiveTransform(CORNERS.reshape(-1, 1, 2), Ht).reshape(4, 2))
    return frames, np.array(truth)


def test_tracks_zoom_pan_rotation_from_middle_frame():
    frames, truth = make_clip()
    ref = 20
    corners, tracked, _ = T.track_corners(frames, ref, truth[ref], smooth=1)
    assert tracked.all()
    err = np.linalg.norm(corners - truth, axis=2).max()
    assert err < 2.0, err


def test_cut_stops_tracking():
    frames, truth = make_clip(30)
    other = [np.full_like(f, (30, 160, 60)) for f in frames[:10]]  # different shot: green scene
    clip = frames[:20] + other
    cuts = detect_cuts(clip)
    assert cuts == [20]
    shot = longest_shot(len(clip), cuts)
    corners, tracked, _ = T.track_corners(clip, 10, truth[10], shot=shot)
    assert np.isnan(corners[20:]).all() and not tracked[20:].any()
    assert tracked[:20].all()


def test_strip_maps_corners_to_strip_edges():
    Hs = T.strip_homography(CORNERS)
    out = cv2.perspectiveTransform(CORNERS.reshape(-1, 1, 2), Hs).reshape(4, 2)
    m = T.STRIP_H / 4
    np.testing.assert_allclose(out, [[0, m], [0, T.STRIP_H - m], [T.STRIP_W, m], [T.STRIP_W, T.STRIP_H - m]], atol=1e-3)


def test_nan_corners_give_black_strip():
    strip = T.warp_strip(np.zeros((H, W, 3), np.uint8), np.full((4, 2), np.nan))
    assert strip.shape == (T.STRIP_H, T.STRIP_W) and strip.max() == 0
