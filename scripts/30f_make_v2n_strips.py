"""CP5 step 2: track the clicked fretboard corners through each clip and warp to 800x144 strips (V2N input).

    python scripts/30f_make_v2n_strips.py

Inputs: data/interim/baseline_clips/shots.json (from 30f_prepare_strip_frames.py) and the clicked corners
data/labels/fretboard_corners/gaps_test_baseline_EVAL_ONLY.jsonl (from 01b_annotate_corners.py).
Writes data/interim/baseline_clips/strips/
    <clip>.npy           (T, 144, 800) uint8 grayscale strips at 25 fps; black where not valid
    <clip>_valid.npy     (T,) bool: frame has a correct fretboard view (inside the clicked shot and not listed
                         in data/interim/baseline_clips/invalid_frames.txt, e.g. dissolves). Evaluation of the
                         baseline is restricted to these frames (30e --mask-dir).
    strips.csv           per clip: coverage (share of frames tracked), lost frames, inliers, skipped
    tracking_sheet.jpg   per clip: 3 frames with the tracked neck (green = tracked, red = held) + their strips
"""

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from guitar2midi.preprocessing import fretboard_tracking as T
from guitar2midi.preprocessing.video import read_frames

CLIPS = Path("data/interim/baseline_clips")
ANN = Path("data/labels/fretboard_corners/gaps_test_baseline_EVAL_ONLY.jsonl")


def block(frames, corners, tracked, strips, idx, tile_w=200):
    """3 frames with the neck quad + the 3 strips below them."""
    tile_h = tile_w * 9 // 16
    top, bottom = [], []
    for t in idx:
        f = frames[t].copy()
        if not np.isnan(corners[t]).any():
            col = (0, 255, 0) if tracked[t] else (0, 0, 255)
            cv2.polylines(f, [T._quad(corners[t]).astype(np.int32)], True, col, max(2, f.shape[1] // 300))
        s = min(tile_w / f.shape[1], tile_h / f.shape[0])
        f = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
        tile = np.zeros((tile_h, tile_w, 3), np.uint8)
        tile[(tile_h - f.shape[0]) // 2:(tile_h - f.shape[0]) // 2 + f.shape[0],
             (tile_w - f.shape[1]) // 2:(tile_w - f.shape[1]) // 2 + f.shape[1]] = f
        top.append(tile)
        bottom.append(cv2.cvtColor(cv2.resize(strips[t], (tile_w, tile_w * T.STRIP_H // T.STRIP_W)), cv2.COLOR_GRAY2BGR))
    return np.vstack([np.hstack(top), np.hstack(bottom)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", type=Path, default=ANN)
    p.add_argument("--out", type=Path, default=CLIPS / "strips")
    p.add_argument("--min-coverage", type=float, default=0.9, help="flag clips with fewer tracked frames")
    args = p.parse_args()

    shots = json.loads((CLIPS / "shots.json").read_text())
    invalid: dict[str, list[tuple[int, int]]] = {}
    inv_file = CLIPS / "invalid_frames.txt"
    if inv_file.exists():
        for ln in inv_file.read_text().splitlines():
            parts = ln.split("#")[0].split()
            if len(parts) == 3:
                invalid.setdefault(parts[0], []).append((int(parts[1]), int(parts[2])))
    anns = {}
    for line in args.annotations.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            anns[r["clip_id"]] = r  # last record per clip wins (re-annotation)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, blocks = [], []
    for cid in sorted(anns):
        r = anns[cid]
        if not r["visible"]:
            rows.append({"clip_id": cid, "skipped": 1})
            print(f"{cid}: skipped (fretboard not usable)")
            continue
        ref = int(re.search(r"f(\d+)\.png$", r["image"]).group(1))
        frames = read_frames(CLIPS / "video" / f"{cid}.mp4")
        corners, tracked, inliers = T.track_corners(frames, ref, np.array(r["corners"]), shot=tuple(shots[cid]["shot"]))
        valid = ~np.isnan(corners[:, 0, 0])
        for a0, b0 in invalid.get(cid, []):
            valid[a0:b0 + 1] = False
        corners[~valid] = np.nan  # black strip where the view is wrong
        strips = np.stack([T.warp_strip(f, c) for f, c in zip(frames, corners)])
        np.save(args.out / f"{cid}.npy", strips)
        np.save(args.out / f"{cid}_valid.npy", valid)
        a, b = shots[cid]["shot"]
        in_shot = inliers[a:b][np.arange(a, b) != ref]
        coverage = float((tracked & valid).mean())
        rows.append({"clip_id": cid, "skipped": 0, "n_frames": len(frames), "ref_frame": ref,
                     "shot_frames": b - a, "coverage": round(float(coverage), 3),
                     "lost_in_shot": int((~tracked[a:b]).sum()),
                     "median_inliers": int(np.median(in_shot)) if len(in_shot) else 0,
                     "min_inliers": int(in_shot.min()) if len(in_shot) else 0,
                     "flag": int(coverage < args.min_coverage)})
        idx = [int(round(x)) for x in np.linspace(0.05, 0.95, 3) * (len(frames) - 1)]
        img = block(frames, corners, tracked, strips, idx)
        label = f"{cid} cov={coverage:.0%} inl={rows[-1]['median_inliers']}" + ("  FLAG" if rows[-1]["flag"] else "")
        lab = np.full((20, img.shape[1], 3), 255, np.uint8)
        cv2.putText(lab, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if rows[-1]["flag"] else (0, 0, 0), 1)
        blocks.append(np.vstack([img, lab]))
        print(f"{cid}: coverage {coverage:.0%}, lost in shot {rows[-1]['lost_in_shot']}, "
              f"median inliers {rows[-1]['median_inliers']}{'  FLAG' if rows[-1]['flag'] else ''}", flush=True)

    fields = max((list(r) for r in rows), key=len)  # skipped rows only have clip_id + skipped
    with open(args.out / "strips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    if blocks:
        per_row, gap = 3, 8
        bw = blocks[0].shape[1] + gap
        bh = max(b.shape[0] for b in blocks)
        n_rows = (len(blocks) + per_row - 1) // per_row
        sheet = np.full((n_rows * bh, per_row * bw, 3), 255, np.uint8)
        for i, b in enumerate(blocks):
            rr, cc = divmod(i, per_row)
            sheet[rr * bh:rr * bh + b.shape[0], cc * bw:cc * bw + b.shape[1]] = b
        cv2.imwrite(str(args.out / "tracking_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    done = [r for r in rows if not r["skipped"]]
    print(f"\n{len(done)} clips with strips, {len(rows) - len(done)} skipped, "
          f"{sum(r['flag'] for r in done)} flagged (coverage < {args.min_coverage:.0%}); see {args.out}")


if __name__ == "__main__":
    main()
