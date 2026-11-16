"""Readers for the public GuiFiR-Vid release (github.com/kmu-ee-ailab/guifir_dataset).

Layout after scripts/01a_prepare_guifir.sh:
    data/raw/guifir/guifir_vid/{train,val,test}/{videos,frames,annotations}/

Each annotation line is one frame with ``filename`` (``<clip>/<clip>_<frame>.png``),
``frame_width/height``, ``frame_tab``, ``neck_annotation`` (detector box) and,
for val/test only, ``frame_gt_pose`` with fretboard corners ``lt/lb/rt/rb`` in
640x360 coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GUIFIR_ROOT = Path("data/raw/guifir/guifir_vid")
GT_POSE_WIDTH = 640  # frame_gt_pose coordinates are in 640x360 space

# Which GuiFiR corner key is (nut/high-E, nut/low-E, f20/high-E, f20/low-E).
# Assumes the usual view: headstock on the image left, low-E string at the top.
# Verify with: scripts/01c_build_fretboard_coco.py --visualize 20
DEFAULT_CORNER_KEYS = ("lb", "lt", "rb", "rt")


def annotation_file(split: str, root: Path = GUIFIR_ROOT) -> Path:
    files = sorted((root / split / "annotations").glob("*.jsonl"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one .jsonl in {root / split / 'annotations'}, found {files}")
    return files[0]


def load_split(split: str, root: Path = GUIFIR_ROOT) -> list[dict]:
    with open(annotation_file(split, root)) as f:
        return [json.loads(line) for line in f if line.strip()]


def clip_id(record: dict) -> str:
    return record["filename"].split("/")[0]


def frame_path(record: dict, split: str, root: Path = GUIFIR_ROOT) -> Path:
    return root / split / "frames" / record["filename"]


def neck_box(record: dict) -> np.ndarray | None:
    """Detector fretboard box [x1, y1, x2, y2] in frame pixels, or None."""
    ann = record.get("neck_annotation") or {}
    boxes = ann.get("boxes")
    if not boxes:
        return None
    box = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return box[0]


def gt_corners(record: dict, keys=DEFAULT_CORNER_KEYS) -> np.ndarray | None:
    """(4, 2) GT corners in frame pixels, in fretboard_grid.CORNER_IDX order."""
    pose = record.get("frame_gt_pose")
    if not pose or not all(k in pose for k in keys):
        return None
    scale = record["frame_width"] / GT_POSE_WIDTH
    return np.array([pose[k][:2] for k in keys], dtype=float) * scale
