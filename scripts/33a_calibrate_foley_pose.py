"""CP15: map our MMPose keypoints to Foley Music's keypoint scale (env guitar2midi).

    python scripts/33a_calibrate_foley_pose.py                      # 1i8XHo25_KU, 30 s from 60 s
    python scripts/33a_calibrate_foley_pose.py --vid IPsA3sz934M --start 30

Foley Music's released keypoints (OpenPose, body65, 22 fps) of a MUSIC acoustic-guitar validation video vs our
MMPose whole-body keypoints of the same YouTube video at the same times (``foley_music_pose``):
1. time alignment: residual for frame shifts -3..+3 (expect 0),
2. linear fit per axis x_theirs = a * x / W + b on points detected in both -> scale/offset for ``normalise``,
   residual per body part (body / left hand / right hand) in their units and as % of the body height,
3. score threshold so that the share of missing points (0, 0, 0) matches theirs per part.
Writes outputs/runs/<date>_cp15_pose_calibration/{calibration.json, overlay.png}.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from guitar2midi.baselines import foley_music_pose as FP

DATA = Path.home() / "external" / "FoleyMusic_data"
PARTS = {"body": slice(0, 25), "left hand": slice(25, 45), "right hand": slice(45, 65)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vid", default="1i8XHo25_KU")
    p.add_argument("--start", type=float, default=60.0)
    p.add_argument("--duration", type=float, default=30.0)
    args = p.parse_args()

    theirs_all = np.load(DATA / "data" / "MUSIC" / "pose" / "acoustic_guitar" / f"{args.vid}.npy")[..., 0]
    i0 = int(round(args.start * FP.FPS))
    est = FP.WholeBodyPose()
    ours, (W, H) = est.video(DATA / "videos" / f"{args.vid}.mp4", args.start, args.duration, progress=True)
    T = len(ours)
    print(f"{args.vid}: ours {ours.shape} at {W}x{H}, theirs {theirs_all.shape} (frames {i0}..{i0 + T})")

    def fit(shift: int):
        th = theirs_all[i0 + shift:i0 + shift + T]
        both = (th[:, 2] > 0) & (ours[:, 2] > 0.3)
        res = {}
        for ax, size in ((0, W), (1, H)):
            x, y = ours[:, ax][both] / size, th[:, ax][both]
            a, b = np.polyfit(x, y, 1)
            res[ax] = (a, b, np.abs(a * x + b - y))
        return th, both, res

    print("time shift (frames) -> median |residual| x / y:")
    best = None
    for s in range(-3, 4):
        if i0 + s < 0 or i0 + s + T > len(theirs_all):
            continue
        _, _, r = fit(s)
        mx, my = np.median(r[0][2]), np.median(r[1][2])
        print(f"  {s:+d}: {mx:.4f} / {my:.4f}")
        if best is None or mx + my < best[1]:
            best = (s, mx + my)
    shift = best[0]
    th, both, r = fit(shift)
    (ax_, bx, _), (ay, by, _) = r[0], r[1]
    print(f"best shift {shift:+d}; x' = {ax_:.4f} * x/W + {bx:.4f};  y' = {ay:.4f} * y/H + {by:.4f}")

    # body height in their units (neck -> mid-hip, x2 as a rough torso+head scale) for relative errors
    torso = np.median(np.linalg.norm(th[:, :2, 1] - th[:, :2, 8], axis=1)[(th[:, 2, 1] > 0) & (th[:, 2, 8] > 0)])
    print(f"their median neck-midhip distance: {torso:.4f}")
    mapped = ours.copy()
    mapped[:, 0] = ax_ * ours[:, 0] / W + bx
    mapped[:, 1] = ay * ours[:, 1] / H + by
    summary = {}
    for name, sl in PARTS.items():
        m = both[:, sl]
        err = np.linalg.norm(mapped[:, :2, sl] - th[:, :2, sl], axis=1)[m]
        miss_theirs = float((th[:, 2, sl] == 0).mean())
        scores = ours[:, 2, sl].ravel()
        thr = float(np.quantile(scores, miss_theirs)) if miss_theirs > 0 else 0.0
        summary[name] = {"median_err": float(np.median(err)), "median_err_rel_torso": float(np.median(err) / torso),
                         "p90_err": float(np.quantile(err, 0.9)), "n_both": int(m.sum()),
                         "missing_share_theirs": miss_theirs, "score_threshold_matching_share": thr}
        print(f"  {name:10s}: median error {np.median(err):.4f} ({np.median(err) / torso:.0%} of neck-midhip), "
              f"p90 {np.quantile(err, 0.9):.4f}; missing theirs {miss_theirs:.0%} -> score threshold {thr:.2f}")

    out = Path("outputs/runs") / f"{date.today().isoformat()}_cp15_pose_calibration"
    out.mkdir(parents=True, exist_ok=True)
    cal = {"vid": args.vid, "start": args.start, "duration": args.duration, "size": [W, H], "shift_frames": shift,
           "scale": [float(ax_), float(ay)], "offset": [float(bx), float(by)], "torso": float(torso), "parts": summary}
    (out / f"calibration_{args.vid}.json").write_text(json.dumps(cal, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k in zip(axes, np.linspace(0, T - 1, 3).astype(int)):
        for arr, c, lab in ((th[k], "tab:red", "Foley Music (OpenPose)"), (mapped[k], "tab:blue", "ours (MMPose)")):
            ok = arr[2] > (0 if arr is th[k] else 0.3)
            ax.scatter(arr[0][ok], arr[1][ok], s=8, c=c, label=lab)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"t = {args.start + k / FP.FPS:.1f} s")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{args.vid}: keypoints in Foley Music's scale")
    fig.tight_layout()
    fig.savefig(out / f"overlay_{args.vid}.png", dpi=100)
    print(f"written {out}")


if __name__ == "__main__":
    main()
