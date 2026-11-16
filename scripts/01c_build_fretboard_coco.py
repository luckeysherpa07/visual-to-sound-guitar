"""Build COCO keypoint files (126 fretboard points) for MMPose.

train: manual corner annotations (scripts/01b_annotate_corners.py)
val:   GuiFiR-Vid val GT corners
test:  GuiFiR-Vid test GT corners (held out; never use for tuning)

The 126 points are generated from 4 corners (fretboard_grid.grid_from_corners).
bbox = tight bbox of the 126 points; the 10% padding of the paper is applied in
the MMPose pipeline (GetBBoxCenterScale padding=1.1). The GuiFiR detector box is
stored as ``neck_box`` for evaluation with realistic boxes.

    python scripts/01c_build_fretboard_coco.py \
        --manual data/labels/fretboard_corners/*.jsonl --visualize 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from guitar2midi.data import guifir
from guitar2midi.preprocessing.fretboard_grid import (
    KEYPOINT_NAMES, corners_valid, draw_grid, grid_from_corners)

DATA_ROOT = Path("data")
OUT_DIR = DATA_ROOT / "processed/fretboard_coco"


def make_entry(image_rel: str, width: int, height: int, corners: np.ndarray,
               label_source: str, neck_box=None) -> dict:
    grid = grid_from_corners(corners)
    x1, y1 = grid.min(0)
    x2, y2 = grid.max(0)
    kps = np.concatenate([grid, np.full((len(grid), 1), 2.0)], 1)
    return dict(image=image_rel, width=width, height=height, keypoints=kps,
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                label_source=label_source,
                neck_box=None if neck_box is None else [float(v) for v in neck_box])


def manual_entries(paths: list[Path], exclude_clips: set[str]) -> list[dict]:
    out = []
    for p in paths:
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if not r["visible"]:
                continue
            if r["clip_id"] in exclude_clips:
                raise ValueError(f"{r['image']}: clip {r['clip_id']} is in GuiFiR val/test")
            if not corners_valid(np.array(r["corners"])):
                print(f"  skip {r['image']}: corners form a crossed/concave quad")
                continue
            out.append(make_entry(r["image"], r["frame_width"], r["frame_height"],
                                  np.array(r["corners"]), r["label_source"]))
    return out


def guifir_entries(split: str, keys) -> list[dict]:
    out, no_frame = [], 0
    for rec in guifir.load_split(split):
        corners = guifir.gt_corners(rec, keys)
        if corners is None:
            continue
        if not guifir.frame_path(rec, split).exists():  # index beyond clip end
            no_frame += 1
            continue
        if not corners_valid(corners):
            print(f"  skip {split} {rec['filename']}: invalid corners (check --corner-keys)")
            continue
        img = guifir.frame_path(rec, split).relative_to(DATA_ROOT)
        out.append(make_entry(str(img), rec["frame_width"], rec["frame_height"],
                              corners, "manual_guifir", guifir.neck_box(rec)))
    if no_frame:
        print(f"  {split}: skipped {no_frame} frames not extracted (see frames/missing.txt)")
    return out


def to_coco(entries: list[dict]) -> dict:
    images, anns = [], []
    for i, e in enumerate(entries, 1):
        images.append(dict(id=i, file_name=e["image"], width=e["width"], height=e["height"]))
        anns.append(dict(
            id=i, image_id=i, category_id=1, iscrowd=0,
            bbox=e["bbox"], area=e["bbox"][2] * e["bbox"][3],
            keypoints=e["keypoints"].reshape(-1).tolist(), num_keypoints=len(e["keypoints"]),
            label_source=e["label_source"], neck_box=e["neck_box"]))
    return dict(images=images, annotations=anns,
                categories=[dict(id=1, name="fretboard", keypoints=KEYPOINT_NAMES, skeleton=[])])


def visualize(entries: list[dict], split: str, n: int):
    out = OUT_DIR / "vis" / split
    out.mkdir(parents=True, exist_ok=True)
    for e in entries[:n]:
        img = cv2.imread(str(DATA_ROOT / e["image"]))
        if img is None:
            continue
        vis = draw_grid(img, e["keypoints"][:, :2], thickness=2)
        if e["neck_box"]:
            x1, y1, x2, y2 = map(int, e["neck_box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.imwrite(str(out / Path(e["image"]).name.replace(".png", ".jpg")), vis)
    print(f"  wrote overlays to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual", type=Path, nargs="*", default=[])
    ap.add_argument("--corner-keys", default=",".join(guifir.DEFAULT_CORNER_KEYS),
                    help="GuiFiR keys for (nut/high-E, nut/low-E, f20/high-E, f20/low-E)")
    ap.add_argument("--visualize", type=int, default=0, help="save N overlay images per split")
    args = ap.parse_args()
    keys = tuple(args.corner_keys.split(","))

    splits = {s: guifir_entries(s, keys) for s in ("val", "test")}
    held_out = {Path(e["image"]).parent.name for s in splits.values() for e in s}
    splits["train"] = manual_entries(args.manual, held_out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, entries in splits.items():
        (OUT_DIR / f"{split}.json").write_text(json.dumps(to_coco(entries)))
        print(f"{split}: {len(entries)} frames")
        if args.visualize:
            visualize(entries, split, args.visualize)


if __name__ == "__main__":
    main()
