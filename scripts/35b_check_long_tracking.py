"""CP17 step 17b check: how far can corners be tracked through a long shot? Measured against manual clicks.

    python scripts/35b_check_long_tracking.py

The GAPS test pieces have up to 3 eval clips with manually clicked corners (CP5) at different times. When two
clicks fall in the same shot of the full video, track from one click to the other's frame and compare with the
manual corners there. Compares the CP5 frame-to-frame tracker (``track_corners``, "lk") with the keyframe
tracker (``track_corners_keyframe``, CP17, "kf<d>": features in the guitar quad dilated by d neck widths). The test clicks are used only to measure the tracker (no training data).

Error = mean distance of the 4 corners / neck length (nut -> fret 20), in % (resolution-free); click noise
between two manual clicks is part of it. Writes outputs/runs/<date>_cp17_long_tracking/{pairs.csv,summary.txt}.
"""

import argparse
import csv
import datetime
import json
import re
from pathlib import Path

import numpy as np

from guitar2midi.baselines import v2n_data as D
from guitar2midi.preprocessing import fretboard_tracking as T
from guitar2midi.preprocessing.video import detect_cuts_file, shots

CLIPS = Path("data/interim/baseline_clips")
ANN = Path("data/labels/fretboard_corners/gaps_test_baseline_EVAL_ONLY.jsonl")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default=f"{datetime.date.today()}_cp17_long_tracking")
    p.add_argument("--max-span-s", type=float, default=240.0, help="skip pairs further apart (memory)")
    p.add_argument("--trackers", nargs="+", default=["lk", "kf1.0"],
                   help="lk = CP5 frame-to-frame; kf<d> = keyframe with features in the guitar quad dilated by d")
    args = p.parse_args()
    out = Path("outputs/runs") / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    with open(CLIPS / "clips.csv") as f:
        clips = {r["clip_id"]: r for r in csv.DictReader(f)}
    anns = {}
    for line in ANN.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["visible"]:
                anns[r["clip_id"]] = r
    by_piece = {}
    for cid, r in anns.items():
        by_piece.setdefault(clips[cid]["piece_id"], []).append(cid)

    rows = []
    for piece in sorted(by_piece):
        cids = by_piece[piece]
        if len(cids) < 2:
            continue
        video = Path("data/raw/gaps/videos") / f"{piece}.mp4"
        w, h = D.video_size(video)
        n, cuts = detect_cuts_file(video)
        sh = shots(n, cuts)
        pts = []
        for cid in cids:
            r = anns[cid]
            t = int(round(float(clips[cid]["start_s"]) * D.FPS)) + int(re.search(r"f(\d+)\.png$", r["image"]).group(1))
            c = np.array(r["corners"], float) * [w / r["frame_width"], h / r["frame_height"]]
            s = next(i for i, (a, b) in enumerate(sh) if a <= t < b)
            pts.append((cid, t, c, s))
        pairs = [(x, y) for x in pts for y in pts if x[0] != y[0] and x[3] == y[3]
                 and abs(x[1] - y[1]) <= args.max_span_s * D.FPS]
        if not pairs:
            print(f"{piece}: no two clicks in the same shot")
            continue
        tw, th = D.track_size(w, h)
        scale = np.array([tw / w, th / h])
        lo, hi = min(min(x[1], y[1]) for x, y in pairs), max(max(x[1], y[1]) for x, y in pairs) + 1
        frames = [f for t, f in enumerate(D.iter_frames(video, (tw, th))) if lo <= t < hi]
        for (ci, ti, c_i, s), (cj, tj, c_j, _) in pairs:
            a, b = min(ti, tj), max(ti, tj) + 1
            seg = frames[a - lo:b - lo]
            ref, tgt = ti - a, tj - a
            res = {"piece": piece, "from": ci, "to": cj, "span_s": round((b - a - 1) / D.FPS, 1)}
            gt = c_j * scale
            nl = T._neck_len(gt)
            for name in args.trackers:
                if name == "lk":
                    cor, tracked, _ = T.track_corners(seg, ref, c_i * scale)
                else:
                    cor, tracked, _ = T.track_corners_keyframe(seg, ref, c_i * scale, feature_dilate=float(name[2:]))
                est = cor[tgt]
                err = np.linalg.norm(est - gt, axis=1).mean() / nl * 100 if np.isfinite(est).all() else np.nan
                res[f"{name}_err_pct"] = round(float(err), 2)
                res[f"{name}_tracked_pct"] = round(float(tracked.mean() * 100), 1)
            rows.append(res)
            print(res, flush=True)

    with open(out / "pairs.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    lines = [f"{len(rows)} click pairs (directed) in the same shot, {len({r['piece'] for r in rows})} pieces, "
             f"span median {np.median([r['span_s'] for r in rows]):.0f} s, max {max(r['span_s'] for r in rows):.0f} s"]
    for name in args.trackers:
        e = np.array([r[f"{name}_err_pct"] for r in rows], float)
        fin = e[np.isfinite(e)]
        lines.append(f"{name}: target frame tracked {len(fin)}/{len(e)}; error (% neck length) median "
                     f"{np.median(fin):.1f}, p90 {np.quantile(fin, .9):.1f}, max {fin.max():.1f}; "
                     f"> 5 %: {(fin > 5).sum()}, > 10 %: {(fin > 10).sum()}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
