from pathlib import Path

import numpy as np
from mmengine.config import Config

from guitar2midi.preprocessing import fretboard_grid as fg

METAINFO = Path(__file__).parents[1] / "configs/preprocessing/fretboard_126_metainfo.py"


def test_sizes():
    assert fg.NUM_KEYPOINTS == 126
    assert len(fg.KEYPOINT_NAMES) == 126
    assert len(fg.CORNER_IDX) == 4
    assert len(fg.EDGE_IDX) == 46
    assert len(fg.INTERNAL_IDX) == 76
    all_idx = np.concatenate([fg.CORNER_IDX, fg.EDGE_IDX, fg.INTERNAL_IDX])
    assert sorted(all_idx.tolist()) == list(range(126))


def test_weights_sum_to_one_per_group():
    for idx in (fg.CORNER_IDX, fg.EDGE_IDX, fg.INTERNAL_IDX):
        assert np.isclose(fg.KEYPOINT_WEIGHTS[idx].sum(), 1.0)


def test_fret_spacing():
    x = fg.fret_positions()
    assert x[0] == 0 and np.isclose(x[-1], 1)
    gaps = np.diff(x)
    assert np.all(gaps > 0) and np.all(np.diff(gaps) < 0)
    # 12th fret is halfway along the scale: d_12 / d_20 = 0.5 / (1 - 2^(-20/12))
    assert np.isclose(x[12], 0.5 / (1 - 2 ** (-20 / 12)))


def test_grid_reproduces_corners():
    corners = np.array([[100, 300], [110, 360], [600, 280], [605, 330]], float)
    grid = fg.grid_from_corners(corners)
    assert grid.shape == (126, 2)
    assert np.allclose(grid[fg.CORNER_IDX], corners, atol=1e-3)


def test_corners_valid():
    good = np.array([[100, 300], [110, 360], [600, 280], [605, 330]], float)
    assert fg.corners_valid(good)
    assert fg.corners_valid(good[:, ::-1])  # mirrored orientation is fine
    crossed = good[[1, 0, 2, 3]]  # nut clicks swapped
    assert not fg.corners_valid(crossed)


def test_corners_from_fret12_clicks_round_trip():
    # perspective quad: clicking the nut and fret 12 of its grid must give back the same corners
    corners = np.array([[492, 43], [480, 16], [278, 137], [266, 108]], float)
    grid = fg.grid_from_corners(corners)
    for fret in (5, 12, 19):
        clicks = grid[[fg.kp_index(0, 0), fg.kp_index(5, 0), fg.kp_index(0, fret), fg.kp_index(5, fret)]]
        assert np.allclose(fg.corners_from_clicks(clicks, fret), corners, atol=1e-2)
    assert np.allclose(fg.corners_from_clicks(corners, 20), corners)


def test_identity_homography_matches_template():
    grid = fg.grid_from_corners(fg.TEMPLATE_CORNERS)
    assert np.allclose(grid, fg.template_grid(), atol=1e-5)


def test_metainfo_matches_grid_module():
    info = Config.fromfile(str(METAINFO)).dataset_info
    names = [info.keypoint_info[i]["name"] for i in range(126)]
    assert names == fg.KEYPOINT_NAMES
    assert np.allclose(info.joint_weights, fg.KEYPOINT_WEIGHTS)
