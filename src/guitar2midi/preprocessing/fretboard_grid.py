"""126-point fretboard grid (6 strings x nut + 20 frets), as in GuiFiR (Sec. III-B).

Index convention: ``idx = string * NUM_FRET_LINES + fret`` with
``string`` 0..5 (0 = high-E / string 1, 5 = low-E / string 6) and
``fret`` 0..20 (0 = nut). Labels are physical, not image-based, so a
mirrored (left-handed) guitar keeps the same labels.

The full grid is generated from the 4 corners (nut and 20th fret on the
outer strings): the corners define a homography from a fronto-parallel
template in which fret n sits at d_n = s * (1 - 2^(-n/12)).
"""

from __future__ import annotations

import cv2
import numpy as np

NUM_STRINGS = 6
NUM_FRETS = 20
NUM_FRET_LINES = NUM_FRETS + 1  # nut + 20 frets
NUM_KEYPOINTS = NUM_STRINGS * NUM_FRET_LINES  # 126

KEYPOINT_NAMES = [
    f"s{s + 1}_f{f:02d}" for s in range(NUM_STRINGS) for f in range(NUM_FRET_LINES)
]


def kp_index(string: int, fret: int) -> int:
    """Grid index for 0-based ``string`` (0 = high E) and ``fret`` (0 = nut)."""
    return string * NUM_FRET_LINES + fret


# Corner order used everywhere: nut/high-E, nut/low-E, f20/high-E, f20/low-E.
CORNER_IDX = np.array(
    [kp_index(0, 0), kp_index(5, 0), kp_index(0, NUM_FRETS), kp_index(5, NUM_FRETS)]
)


def _point_groups() -> tuple[np.ndarray, np.ndarray]:
    edge, internal = [], []
    corners = set(CORNER_IDX.tolist())
    for s in range(NUM_STRINGS):
        for f in range(NUM_FRET_LINES):
            i = kp_index(s, f)
            if i in corners:
                continue
            on_border = s in (0, NUM_STRINGS - 1) or f in (0, NUM_FRETS)
            (edge if on_border else internal).append(i)
    return np.array(edge), np.array(internal)


EDGE_IDX, INTERNAL_IDX = _point_groups()  # 46 and 76 points

# Loss weights from the paper: each group sums to 1.
KEYPOINT_WEIGHTS = np.empty(NUM_KEYPOINTS, dtype=np.float64)
KEYPOINT_WEIGHTS[CORNER_IDX] = 1.0 / len(CORNER_IDX)
KEYPOINT_WEIGHTS[EDGE_IDX] = 1.0 / len(EDGE_IDX)
KEYPOINT_WEIGHTS[INTERNAL_IDX] = 1.0 / len(INTERNAL_IDX)


def fret_positions(num_frets: int = NUM_FRETS) -> np.ndarray:
    """Normalised distance of nut (0) .. fret ``num_frets`` (1) from the nut."""
    n = np.arange(num_frets + 1)
    d = 1.0 - 2.0 ** (-n / 12.0)
    return d / d[-1]


def template_grid() -> np.ndarray:
    """(126, 2) grid in a unit rectangle: x = fret position, y = string (0..1)."""
    xs = fret_positions()
    ys = np.linspace(0.0, 1.0, NUM_STRINGS)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)  # (6, 21, 2), [string, fret]
    return grid.reshape(-1, 2)


TEMPLATE_CORNERS = template_grid()[CORNER_IDX].astype(np.float32)


def corners_valid(corners: np.ndarray) -> bool:
    """True if the corners form a convex, non-self-intersecting quad.

    Crossed or concave quads (e.g. swapped clicks) give a degenerate homography
    that throws grid points far outside the image.
    """
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    quad = c[[0, 1, 3, 2]]  # walk the boundary: nut/hi, nut/lo, f20/lo, f20/hi
    edges = np.roll(quad, -1, axis=0) - quad
    nxt = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]
    return bool(np.all(cross > 0) or np.all(cross < 0))


def grid_from_corners(corners: np.ndarray) -> np.ndarray:
    """Full (126, 2) image-space grid from 4 corners.

    ``corners``: (4, 2) in CORNER_IDX order
    (nut/high-E, nut/low-E, f20/high-E, f20/low-E).
    """
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    H = cv2.getPerspectiveTransform(TEMPLATE_CORNERS, corners)
    pts = template_grid().reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.float64)


def corners_from_clicks(clicks: np.ndarray, fret: int = NUM_FRETS) -> np.ndarray:
    """The 4 standard corners (nut + fret 20) from clicks on the nut and fret ``fret``.

    ``clicks``: (4, 2) as nut/high-E, nut/low-E, fret/high-E, fret/low-E. Used when
    fret 20 is hidden or missing (classical guitars have 19 frets): the user clicks
    e.g. fret 12 and fret 20 is extrapolated with the same fret formula / homography
    that generates the grid, so the result is exact up to click noise.
    """
    clicks = np.asarray(clicks, dtype=np.float32).reshape(4, 2)
    if fret == NUM_FRETS:
        return clicks.astype(np.float64)
    if not 1 <= fret < NUM_FRETS:
        raise ValueError(f"fret must be in 1..{NUM_FRETS}, got {fret}")
    x = fret_positions()[fret]
    src = np.array([[0, 0], [0, 1], [x, 0], [x, 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, clicks)
    return cv2.perspectiveTransform(TEMPLATE_CORNERS.reshape(-1, 1, 2), H).reshape(4, 2).astype(np.float64)


# Bridge in template units (nut = 0, fret 20 = 1): the bridge is at the full scale length, fret n at
# s * (1 - 2^(-n/12)), so bridge / fret20 = 1 / (1 - 2^(-20/12)) = 1.4598.
BRIDGE_X = 1.0 / (1.0 - 2.0 ** (-NUM_FRETS / 12.0))


def extend_corners(corners: np.ndarray, x_end: float = BRIDGE_X) -> np.ndarray:
    """Corners (nut + fret 20) -> (nut, position ``x_end``) along the strings, same corner order.

    ``x_end`` in template units (fret 20 = 1, bridge = ``BRIDGE_X``). Uses the homography of the 4 corners,
    i.e. the neck plane is extrapolated beyond fret 20 (over the body to the bridge). Exact for a flat
    plane up to click/tracking noise; the string fan-out beyond fret 20 follows the same projective taper.
    Returns NaN corners if the input is NaN or the extension is degenerate (not a convex quad).
    """
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if np.isnan(c).any():
        return np.full((4, 2), np.nan)
    H = cv2.getPerspectiveTransform(TEMPLATE_CORNERS, c.astype(np.float32))
    src = np.array([[0, 0], [0, 1], [x_end, 0], [x_end, 1]], dtype=np.float32)
    out = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(4, 2).astype(np.float64)
    return out if np.isfinite(out).all() and corners_valid(out) else np.full((4, 2), np.nan)


def draw_grid(image: np.ndarray, grid: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Draw strings (green), frets (blue), nut (red) and corners on a copy of ``image``."""
    out = image.copy()
    g = np.round(grid.reshape(NUM_STRINGS, NUM_FRET_LINES, 2)).astype(int)
    for s in range(NUM_STRINGS):
        cv2.polylines(out, [g[s]], False, (0, 200, 0), thickness)
    for f in range(NUM_FRET_LINES):
        color = (0, 0, 255) if f == 0 else (255, 120, 0)
        cv2.polylines(out, [g[:, f]], False, color, thickness)
    for i, c in zip(CORNER_IDX, [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]):
        cv2.circle(out, tuple(np.round(grid[i]).astype(int)), 4, c, -1)
    return out
