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
# lt/lb/rt/rb are IMAGE sides (lt.x < rt.x in 2,589/2,591 val+test frames).
# The player faces the camera, so the nut is on the image right: in all 1,483
# low-fret frames the fretting hand is in the right half, and the right side is
# narrower. Low E is at the top. Checked visually on val overlays (2026-11-16).
DEFAULT_CORNER_KEYS = ("rb", "rt", "lb", "lt")


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


def frame_index(record: dict) -> int:
    """0-based native-fps frame index encoded in ``<clip>_NNN.png``."""
    return int(Path(record["filename"]).stem.rsplit("_", 1)[1])


def extract_annotated_frames(split: str, root: Path = GUIFIR_ROOT) -> tuple[int, int]:
    """Save only the annotated frames of each clip as ``frames/<clip>/<clip>_NNN.png``.

    Reads each video once, sequentially (no seeking). Indices beyond the end of
    the clip (18 clips annotated at a different fps / off by one) are listed in
    ``frames/missing.txt``. Returns (saved, missing).
    """
    import cv2
    from collections import defaultdict

    frames_dir = root / split / "frames"
    wanted = defaultdict(dict)  # clip -> {frame index: output path}
    for rec in load_split(split, root):
        wanted[clip_id(rec)][frame_index(rec)] = frames_dir / rec["filename"]

    saved, missing = 0, []
    for cid in sorted(wanted):
        (frames_dir / cid).mkdir(parents=True, exist_ok=True)
        todo = {i: p for i, p in wanted[cid].items() if not p.exists()}
        saved += len(wanted[cid]) - len(todo)
        if not todo:
            continue
        cap = cv2.VideoCapture(str(root / split / "videos" / f"{cid}.mp4"))
        i, last = 0, max(todo)
        while i <= last:
            ok, img = cap.read()
            if not ok:
                break
            if i in todo:
                cv2.imwrite(str(todo.pop(i)), img)
                saved += 1
            i += 1
        cap.release()
        missing += [p.name for p in todo.values()]
    (frames_dir / "missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    return saved, len(missing)


def gt_corners(record: dict, keys=DEFAULT_CORNER_KEYS) -> np.ndarray | None:
    """(4, 2) GT corners in frame pixels, in fretboard_grid.CORNER_IDX order."""
    pose = record.get("frame_gt_pose")
    if not pose or not all(k in pose for k in keys):
        return None
    scale = record["frame_width"] / GT_POSE_WIDTH
    return np.array([pose[k][:2] for k in keys], dtype=float) * scale
