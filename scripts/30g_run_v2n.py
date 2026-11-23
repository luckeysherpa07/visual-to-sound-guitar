"""CP6: V2N zero-shot (public PianoVAM checkpoint) on the guitar fretboard strips (env ``v2n``).

    conda activate v2n
    python scripts/30g_run_v2n.py                 # all clips with strips
    python scripts/30g_run_v2n.py --limit 2       # quick test

Input:  data/interim/baseline_clips/strips/<clip>.npy (from 30f_make_v2n_strips.py)
Writes: data/interim/baseline_clips/transcriptions/v2n_zeroshot/<clip>.mid   (scored by 30e like any system)
        data/interim/baseline_clips/v2n_probs/<clip>.npz                     (stitched onset/offset/frame probs)
        data/interim/baseline_clips/plots/v2n/<clip>.png                     (GT vs V2N roll, --plots clips)
Then:   python scripts/30e_evaluate_baselines.py --run-name cp7_v2n --mask-dir data/interim/baseline_clips/strips
"""

import argparse
import time
from pathlib import Path

import numpy as np

from guitar2midi.baselines import v2n

CLIPS = Path("data/interim/baseline_clips")
SYSTEM = "v2n_zeroshot"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strips", type=Path, default=CLIPS / "strips")
    p.add_argument("--checkpoint", type=Path, default=v2n.V2N_CKPT)
    p.add_argument("--limit", type=int, help="only the first N clips (testing)")
    p.add_argument("--plots", nargs="*", default=["019_Vpswc_c1", "212_y41wc_c2", "291_3Sswc_c3"],
                   help="clips to plot (missing ones are skipped)")
    args = p.parse_args()

    clips = sorted(f.stem for f in args.strips.glob("*.npy") if not f.stem.endswith("_valid"))
    if args.limit:
        clips = clips[:args.limit]
    out_midi = CLIPS / "transcriptions" / SYSTEM
    out_probs = CLIPS / "v2n_probs"
    out_probs.mkdir(parents=True, exist_ok=True)

    model = v2n.V2NTranscriber(args.checkpoint)
    print(f"V2N {args.checkpoint.name} on {model.device}, repo {v2n.V2N_REPO}; {len(clips)} clips")
    t0 = time.time()
    n_notes = []
    for clip in clips:
        strips = np.load(args.strips / f"{clip}.npy")
        probs = model.probs(strips)
        notes = model.decode(probs)
        v2n.write_midi(notes, out_midi / f"{clip}.mid")
        np.savez_compressed(out_probs / f"{clip}.npz", **{k: v.astype(np.float16) for k, v in probs.items()})
        n_notes.append(len(notes))
        if clip in args.plots:
            valid_f = args.strips / f"{clip}_valid.npy"
            v2n.plot_roll(strips, probs, notes, CLIPS / "midi" / f"{clip}.mid",
                          np.load(valid_f) if valid_f.exists() else None,
                          CLIPS / "plots" / "v2n" / f"{clip}.png", title=f"{clip}: V2N zero-shot")
        on = probs["onset_probs"]
        print(f"{clip}: {len(notes):3d} notes, max onset prob {on.max():.2f}, "
              f"frames>0.5 {(probs['frame_probs'] > 0.5).mean():.3%}")
    print(f"done in {time.time() - t0:.0f} s; notes per clip: median {np.median(n_notes):.0f}, "
          f"zero-note clips {sum(n == 0 for n in n_notes)}/{len(n_notes)} -> {out_midi}")


if __name__ == "__main__":
    main()
