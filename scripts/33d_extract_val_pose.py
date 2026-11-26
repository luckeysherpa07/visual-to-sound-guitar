"""CP15: our MMPose keypoints for Foley Music's own MUSIC acoustic-guitar validation videos (env guitar2midi).

    python scripts/33d_extract_val_pose.py            # the 4 downloaded val videos, first 120 s each

Same pipeline as for our clips (``foley_music_pose.clip_pose``), frame-aligned with Foley Music's released arrays
(their frame i = video time (i - 2) / 22 s, see 33a). Used by 33e_foley_val_nll.py to compare the trained model's
validation loss with their keypoints vs ours on the same windows.
Writes ~/external/FoleyMusic_data/pose_ours/<vid>.npy (T, 3, 65, 1).
"""

import argparse
from pathlib import Path

import numpy as np

from guitar2midi.baselines import foley_music_pose as FP

DATA = Path.home() / "external" / "FoleyMusic_data"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vids", nargs="*", default=["1i8XHo25_KU", "IPsA3sz934M", "HaX5E66eW24", "3AQsq1PG1Wc"])
    p.add_argument("--duration", type=float, default=120.0)
    args = p.parse_args()

    est = FP.WholeBodyPose()
    out = DATA / "pose_ours"
    out.mkdir(exist_ok=True)
    for v in args.vids:
        pose = FP.clip_pose(est, DATA / "videos" / f"{v}.mp4", args.duration)
        np.save(out / f"{v}.npy", pose.astype(np.float32))
        print(f"{v}: {pose.shape}, detected body {(pose[:, 2, :25] > 0).mean():.0%} hands "
              f"{(pose[:, 2, 25:] > 0).mean():.0%}", flush=True)


if __name__ == "__main__":
    main()
