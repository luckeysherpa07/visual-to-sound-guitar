"""CP12: full-string strips (nut -> bridge), so the picking hand is in the strip (env guitar2midi).

    python scripts/30j_make_full_strips.py --clips-dir data/interim/baseline_clips            # GAPS
    python scripts/30j_make_full_strips.py --clips-dir data/interim/baseline_clips_guifir
    python scripts/30j_make_full_strips.py --clips-dir data/interim/baseline_clips_magcil

Reads the tracked corners of the fretboard strips (<clips-dir>/strips/<clip>_corners.npy, nut + fret 20, written by
30f_make_v2n_strips.py / 31a_make_guifir_clips.py) and extends them along the strings to the bridge with the fret
formula (``fretboard_grid.extend_corners``: bridge = 1.46 x the nut-fret-20 distance). Same tracking, no new
clicks, same strip format (800x144 grayscale, nut left, high E on top, strings in the middle half of the height).
The fret-20 end now lies at 800 / 1.46 = 548 px; the rest shows the body, soundhole/pickups and the picking hand.

Writes <clips-dir>/strips_full/
    <clip>.npy            (T, 144, 800) uint8; black where the frame is not valid (same as the fretboard strips)
    <clip>_valid.npy      copy of strips/<clip>_valid.npy (same frames are scored for both strip versions)
    strips.csv            per clip: share of valid frames whose bridge corners lie inside the image, black share
    tracking_sheet.jpg    per clip 3 frames: fretboard quad (green) + full-string quad (blue), and both strips
The bridge part is extrapolated beyond the clicked points, so it is less precise than the fretboard part; parts
outside the image come out black.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from guitar2midi.preprocessing import fretboard_tracking as T
from guitar2midi.preprocessing.fretboard_grid import extend_corners
from guitar2midi.preprocessing.video import read_frames


def sheet_block(frames, corners, full, strips, strips_full, idx, label, tile_w=240):
    tile_h = tile_w * 9 // 16
    cols = []
    for t in idx:
        f = frames[t].copy()
        th = max(2, f.shape[1] // 300)
        if not np.isnan(full[t]).any():
            cv2.polylines(f, [T._quad(full[t]).astype(np.int32)], True, (255, 128, 0), th)
        if not np.isnan(corners[t]).any():
            cv2.polylines(f, [T._quad(corners[t]).astype(np.int32)], True, (0, 255, 0), th)
        s = min(tile_w / f.shape[1], tile_h / f.shape[0])
        f = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
        tile = np.zeros((tile_h, tile_w, 3), np.uint8)
        tile[(tile_h - f.shape[0]) // 2:(tile_h - f.shape[0]) // 2 + f.shape[0],
             (tile_w - f.shape[1]) // 2:(tile_w - f.shape[1]) // 2 + f.shape[1]] = f
        sh = tile_w * T.STRIP_H // T.STRIP_W
        a = cv2.cvtColor(cv2.resize(strips[t], (tile_w, sh)), cv2.COLOR_GRAY2BGR)
        b = cv2.cvtColor(cv2.resize(strips_full[t], (tile_w, sh)), cv2.COLOR_GRAY2BGR)
        cols.append(np.vstack([tile, a, np.full((2, tile_w, 3), 255, np.uint8), b]))
    img = np.hstack(cols)
    lab = np.full((20, img.shape[1], 3), 255, np.uint8)
    cv2.putText(lab, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, lab])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, required=True)
    args = p.parse_args()

    src, out = args.clips_dir / "strips", args.clips_dir / "strips_full"
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted(f.stem[: -len("_corners")] for f in src.glob("*_corners.npy"))
    rows, blocks = [], []
    for cid in clips:
        corners = np.load(src / f"{cid}_corners.npy").astype(np.float64)
        valid = np.load(src / f"{cid}_valid.npy").astype(bool)
        strips = np.load(src / f"{cid}.npy")
        frames = read_frames(args.clips_dir / "video" / f"{cid}.mp4")
        h, w = frames[0].shape[:2]
        full = np.stack([extend_corners(c) for c in corners])
        full[~valid[: len(full)]] = np.nan
        strips_full = np.stack([T.warp_strip(f, c) for f, c in zip(frames, full)])
        np.save(out / f"{cid}.npy", strips_full)
        np.save(out / f"{cid}_valid.npy", valid)

        ok = ~np.isnan(full[:, 0, 0])
        bridge = full[ok][:, 2:]  # fret-end corners = bridge corners
        inside = ((bridge[..., 0] >= 0) & (bridge[..., 0] < w) & (bridge[..., 1] >= 0) & (bridge[..., 1] < h)).all(1)
        degenerate = int((valid[: len(full)] & ~ok).sum())
        black = float((strips_full[ok] == 0).mean()) if ok.any() else 1.0
        rows.append({"clip_id": cid, "valid_frames": int(valid.sum()), "degenerate_frames": degenerate,
                     "bridge_in_frame": round(float(inside.mean()) if len(inside) else 0.0, 3),
                     "black_share": round(black, 3)})
        idx = [int(round(x)) for x in np.linspace(0.05, 0.95, 3) * (len(frames) - 1)]
        idx = [i if valid[i] else int(np.flatnonzero(valid)[np.argmin(np.abs(np.flatnonzero(valid) - i))]) for i in idx]
        label = f"{cid}  bridge in frame {rows[-1]['bridge_in_frame']:.0%}  black {black:.0%}"
        blocks.append(sheet_block(frames, corners, full, strips, strips_full, idx, label))
        print(f"{cid}: bridge in frame {rows[-1]['bridge_in_frame']:.0%} of valid frames, black {black:.0%}, "
              f"degenerate {degenerate}", flush=True)

    with open(out / "strips.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    T.save_sheet(blocks, out / "tracking_sheet.jpg")
    bif = np.array([r["bridge_in_frame"] for r in rows])
    print(f"\n{len(rows)} clips -> {out}; bridge inside the image: median {np.median(bif):.0%} of frames, "
          f"{(bif < 0.5).sum()} clips < 50 %; black share median {np.median([r['black_share'] for r in rows]):.0%}; "
          f"degenerate frames {sum(r['degenerate_frames'] for r in rows)}")


if __name__ == "__main__":
    main()
