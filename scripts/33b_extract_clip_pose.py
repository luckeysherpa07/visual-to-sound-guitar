"""CP15: body + hand keypoints of our eval clips in Foley Music's format (env guitar2midi).

    python scripts/33b_extract_clip_pose.py                                        # GAPS
    python scripts/33b_extract_clip_pose.py --clips-dir data/interim/baseline_clips_guifir

Input:  <clips-dir>/video/<clip>.mp4 (silent eval clips), length from clips.csv ``length_s`` (default 8 s)
Model:  MMPose whole-body (RTMPose-m + RTMDet-m) -> body65, 22 fps, Foley Music's scale and 2-frame lag
        (``foley_music_pose``, calibrated by 33a_calibrate_foley_pose.py)
Writes: <clips-dir>/pose_body65/<clip>.npy   (T, 3, 65, 1) float32, T = round(length * 22)
        <clips-dir>/plots/pose_body65_sheet.jpg  keypoints drawn on 3 frames of --sheet clips (visual check)
Only clips with a fretboard strip are processed (the masked evaluation set), unless --all.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from guitar2midi.baselines import foley_music_pose as FP

BODY_EDGES = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (8, 12), (12, 13), (1, 0)]


def draw(img, pose, size):
    """pose (3, 65) in Foley scale -> drawn on the BGR image (back to pixels)."""
    import cv2

    W, H = size
    ok = pose[2] > 0
    px = np.stack([(pose[0] - FP.OFFSET[0]) / FP.SCALE[0] * W, (pose[1] - FP.OFFSET[1]) / FP.SCALE[1] * H], 1)
    for a, b in BODY_EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(map(int, px[a])), tuple(map(int, px[b])), (0, 200, 255), 2)
    for i in np.flatnonzero(ok):
        c = (0, 255, 0) if i < 25 else (255, 0, 0) if i < 45 else (0, 0, 255)  # body / left hand / right hand
        cv2.circle(img, tuple(map(int, px[i])), 3 if i < 25 else 2, c, -1)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=Path("data/interim/baseline_clips"))
    p.add_argument("--all", action="store_true", help="all kept clips, not only those with strips")
    p.add_argument("--sheet", type=int, default=6, help="clips in the check sheet")
    args = p.parse_args()

    import cv2

    d = args.clips_dir
    rows = [r for r in csv.DictReader(open(d / "clips.csv")) if r.get("keep", "1") == "1"]
    if not args.all:
        rows = [r for r in rows if (d / "strips" / f"{r['clip_id']}.npy").exists()]
    out = d / "pose_body65"
    out.mkdir(exist_ok=True)
    est = FP.WholeBodyPose()
    sheet_clips = {r["clip_id"] for r in rows[:: max(len(rows) // args.sheet, 1)][:args.sheet]}
    tiles, stats = [], []
    for r in rows:
        c = r["clip_id"]
        length = float(r.get("length_s") or 8.0)
        video = d / "video" / f"{c}.mp4"
        pose = FP.clip_pose(est, video, length)[..., 0]  # (T, 3, 65)
        np.save(out / f"{c}.npy", pose[..., None].astype(np.float32))
        det = pose[:, 2] > 0
        stats.append((det[:, :25].mean(), det[:, 25:45].mean(), det[:, 45:].mean()))
        print(f"{c}: {len(pose)} frames, detected body {stats[-1][0]:.0%} left hand {stats[-1][1]:.0%} "
              f"right hand {stats[-1][2]:.0%}", flush=True)
        if c in sheet_clips:
            cap = cv2.VideoCapture(str(video))
            W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            row = []
            for k in np.linspace(FP.LAG_FRAMES, len(pose) - 1, 3).astype(int):
                cap.set(cv2.CAP_PROP_POS_MSEC, (k - FP.LAG_FRAMES) / FP.FPS * 1000)
                ok, img = cap.read()
                img = draw(img, pose[k], (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
                row.append(cv2.resize(img, (320, int(320 * H / W))))
            cap.release()
            h = min(t.shape[0] for t in row)
            strip = np.concatenate([t[:h] for t in row], 1)
            cv2.putText(strip, c, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(strip)
    s = np.array(stats)
    print(f"{len(rows)} clips -> {out}; detected share mean: body {s[:, 0].mean():.0%}, left hand {s[:, 1].mean():.0%}, "
          f"right hand {s[:, 2].mean():.0%}")
    if tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]
        (d / "plots").mkdir(exist_ok=True)
        cv2.imwrite(str(d / "plots" / "pose_body65_sheet.jpg"), np.concatenate(tiles, 0))


if __name__ == "__main__":
    main()
