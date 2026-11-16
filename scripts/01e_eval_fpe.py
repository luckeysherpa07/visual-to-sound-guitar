"""Evaluate the fretboard pose estimator on GuiFiR-Vid test (or val).

Metrics, in pixels at 640x360 like the GuiFiR paper:
    FCD = mean distance of the 4 predicted corners to GT (paper's metric)
    MPE = mean distance over all 126 points

    python scripts/01e_eval_fpe.py --run outputs/runs/<run> --boxes detector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from guitar2midi.data.guifir import GT_POSE_WIDTH
from guitar2midi.preprocessing.fpe_inference import FretboardPoseEstimator
from guitar2midi.preprocessing.fretboard_grid import CORNER_IDX, draw_grid

DATA_ROOT = Path("data")
COCO_DIR = DATA_ROOT / "processed/fretboard_coco"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run folder with config + best checkpoint")
    ap.add_argument("--checkpoint", type=Path, help="default: best_*.pth in --run")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--boxes", default="detector", choices=["detector", "gt"],
                    help="detector = GuiFiR neck boxes (realistic), gt = boxes from GT points")
    ap.add_argument("--save-vis", type=int, default=30)
    args = ap.parse_args()

    config = next(args.run.glob("*.py"))
    ckpt = args.checkpoint or sorted(args.run.glob("best_*.pth"))[-1]
    fpe = FretboardPoseEstimator(str(config), str(ckpt))

    coco = json.loads((COCO_DIR / f"{args.split}.json").read_text())
    images = {im["id"]: im for im in coco["images"]}
    vis_dir = args.run / f"eval_{args.split}_{args.boxes}"
    vis_dir.mkdir(exist_ok=True)

    fcd, mpe, skipped = [], [], 0
    for n, ann in enumerate(coco["annotations"]):
        im = images[ann["image_id"]]
        if args.boxes == "detector":
            if not ann["neck_box"]:
                skipped += 1
                continue
            box = np.array(ann["neck_box"])
        else:
            x, y, w, h = ann["bbox"]
            box = np.array([x, y, x + w, y + h])
        img = cv2.imread(str(DATA_ROOT / im["file_name"]))
        pred = fpe.predict(img, box[None])[0]
        gt = np.array(ann["keypoints"]).reshape(-1, 3)[:, :2]
        err = np.linalg.norm(pred - gt, axis=1) * GT_POSE_WIDTH / im["width"]
        fcd.append(err[CORNER_IDX].mean())
        mpe.append(err.mean())
        if n < args.save_vis:
            cv2.imwrite(str(vis_dir / f"{n:04d}.jpg"), draw_grid(img, pred, thickness=2))

    res = dict(split=args.split, boxes=args.boxes, checkpoint=str(ckpt), n=len(fcd),
               skipped_no_box=skipped, FCD=float(np.mean(fcd)), MPE=float(np.mean(mpe)),
               FCD_median=float(np.median(fcd)))
    print(json.dumps(res, indent=2))
    (vis_dir / "metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
