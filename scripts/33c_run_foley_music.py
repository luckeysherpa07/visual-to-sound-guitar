"""CP15: Foley Music (retrained on MUSIC acoustic guitar) on our eval clips (env foleymusic).

    PYTHONNOUSERSITE=1 python scripts/33c_run_foley_music.py                         # GAPS
    PYTHONNOUSERSITE=1 python scripts/33c_run_foley_music.py --clips-dir data/interim/baseline_clips_guifir

Input:  <clips-dir>/pose_body65/<clip>.npy (33b_extract_clip_pose.py), clip length from clips.csv ``length_s``
Model:  ``guitar2midi.baselines.foley_music.FoleyMusic`` (default: the CP15 acoustic-guitar run, best val loss)
Writes: <clips-dir>/transcriptions/foleymusic/<clip>.mid   (scored by 30e like any system)
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from guitar2midi.baselines import foley_music as FM


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=Path("data/interim/baseline_clips"))
    p.add_argument("--exp", type=Path, default=FM.EXP)
    p.add_argument("--checkpoint", default="model_best.pth.tar")
    p.add_argument("--system", default="foleymusic")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    d = args.clips_dir
    lengths = {r["clip_id"]: float(r.get("length_s") or 8.0) for r in csv.DictReader(open(d / "clips.csv"))}
    clips = sorted(f.stem for f in (d / "pose_body65").glob("*.npy"))[:args.limit]
    model = FM.FoleyMusic(args.exp, args.checkpoint, beam=args.beam)
    print(f"Foley Music {args.exp}/{args.checkpoint}, window {model.window} frames, beam {args.beam}; {len(clips)} clips")
    out = d / "transcriptions" / args.system
    t0, n = time.time(), []
    for c in clips:
        notes = model.transcribe(np.load(d / "pose_body65" / f"{c}.npy"), lengths[c])
        FM.write_midi(notes, out / f"{c}.mid")
        n.append(len(notes))
        pitches = [x[2] for x in notes]
        print(f"{c}: {len(notes):3d} notes" + (f", pitch {min(pitches)}-{max(pitches)}" if notes else ""), flush=True)
    print(f"done in {time.time() - t0:.0f} s; notes per clip median {np.median(n):.0f}, zero-note clips "
          f"{sum(x == 0 for x in n)}/{len(n)} -> {out}")


if __name__ == "__main__":
    main()
