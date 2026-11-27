"""CP17: our V2N training-data writer (guitar2midi.baselines.v2n_data) against V2N's own label / dataset code.

The V2N parts are skipped unless the V2N repo is importable (run with the v2n env:
``~/miniforge3/envs/v2n/bin/python -m pytest tests/test_v2n_data.py``).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from guitar2midi.baselines import v2n_data as D

V2N_REPO = Path(os.environ.get("V2N_REPO", Path.home() / "external" / "V2N"))


def test_valid_runs_bridges_short_gaps_and_drops_edges():
    tracked = np.ones(200, bool)
    tracked[60:63] = False      # 3-frame loss: bridged
    tracked[120:140] = False    # 20-frame loss: splits
    runs = D.valid_runs(tracked, max_gap=5, edge=2, min_len=50)
    assert runs == [(2, 120), (140, 198)]
    assert D.valid_runs(tracked[:40], min_len=50) == []


def test_record_notes_shift_and_overlap():
    notes = np.array([[1.0, 2.0, 60], [4.9, 5.5, 62], [9.0, 9.5, 64]])  # GAPS time
    rel = D.record_notes(notes, lag=0.1, t0=5.0, t1=8.0)  # video = GAPS + 0.1
    # note 1 ends at 2.1 (before t0): dropped; note 2 starts at 5.0 -> 0.0; note 3 at 9.1 >= t1: dropped
    assert rel.shape == (1, 3)
    np.testing.assert_allclose(rel[0], [0.0, 0.6, 62], atol=1e-9)


def _v2n():
    if not (V2N_REPO / "data" / "midi.py").exists():
        pytest.skip("V2N repo not found")
    pytest.importorskip("hydra")
    pytest.importorskip("mido")
    if str(V2N_REPO) not in sys.path:
        sys.path.insert(0, str(V2N_REPO))


def test_tsv_roundtrip_with_v2n_labeler(tmp_path):
    _v2n()
    from data.midi import LinearLabeler

    rel = np.array([[0.40, 1.00, 64], [-0.04, 0.50, 40], [0.80, 0.84, 83]])  # 2nd: onset 1 frame before start
    tsv = tmp_path / "r.tsv"
    D.write_tsv(rel, tsv)
    lab = LinearLabeler(fps=25.0, n_keys=88, min_midi=21, lenience=2, smooth_keys=["onset", "frame"])
    label, _, _ = lab(str(tsv), 50)
    onset, frame = label[:, :, 0].numpy(), label[:, :, 1].numpy()
    assert onset[10, 64 - 21] == 1.0 and onset[9, 64 - 21] == 0.5 and onset[12, 64 - 21] == 0.0
    assert frame[10:25, 64 - 21].min() == 1.0 and frame[25:, 64 - 21].max() == 0.0
    assert frame[0:12, 40 - 21].min() > 0 and onset[0, 40 - 21] == 0.5  # started before the record
    assert onset[20, 83 - 21] == 1.0


def test_hdf5_record_readable_by_vptdataset(tmp_path):
    _v2n()
    import cv2
    import h5py
    from omegaconf import OmegaConf

    from data.dataset import VPTDataset

    n = 60
    root = tmp_path / "fret"
    piece = root / "pieces" / "p.h5"
    piece.parent.mkdir(parents=True)
    tsv = tmp_path / "TSV" / "p_s0_r0.tsv"
    D.write_tsv(np.array([[0.4, 1.0, 64]]), tsv)
    vlen = h5py.special_dtype(vlen=np.dtype("uint8"))
    with h5py.File(piece, "w") as f:
        g = f.create_group("p_s0_r0")
        ds = g.create_dataset("frames", (n,), dtype=vlen)
        for t in range(n):
            img = np.full((144, 800), t * 4, np.uint8)
            ds[t] = np.frombuffer(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), np.uint8)
        g.attrs.update({"n_frames": n, "duration": n / 25, "actual_fps": 25.0, "tsv_path": str(tsv)})
    D.build_master({"p_s0_r0": piece}, root / "processed_v2" / "train" / D.H5_NAME)

    cfg = OmegaConf.create({"data_dir": str(root), "dataset": {
        "data_dir": "processed_v2", "hdf5_filename": D.H5_NAME, "fps": 25, "min_midi": 21, "max_midi": 108,
        "train": {"split": ["train"], "coverage_seconds": None, "segment_seconds": 1.0, "segment_stride_frames": 1,
                  "labeler": {"_target_": "data.midi.LinearLabeler", "lenience": 2,
                              "smooth_keys": ["onset", "frame"]}}}})
    ds = VPTDataset(cfg, split_key="train")
    assert len(ds) == n - 25 + 1
    item = ds[0]
    assert tuple(item["pixel_values"].shape) == (25, 144, 800)
    assert tuple(item["labels"]["onset"].shape) == (25, 88)
    assert abs(float(item["pixel_values"][5].mean()) - 20 / 255) < 0.02  # frame 5 = gray level 20
    assert item["labels"]["onset"][10, 64 - 21] == 1.0
