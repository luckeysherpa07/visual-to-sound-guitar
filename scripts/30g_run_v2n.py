"""CP6/CP10: V2N zero-shot (public PianoVAM checkpoint) on the guitar fretboard strips (env ``v2n``).

    conda activate v2n
    python scripts/30g_run_v2n.py                 # GAPS, all clips with strips
    python scripts/30g_run_v2n.py --limit 2       # quick test
    python scripts/30g_run_v2n.py --clips-dir data/interim/baseline_clips_guifir     # other clip sets (CP10)
    python scripts/30g_run_v2n.py --flip v --layout top                              # strip variant (CP10 re-check)

Input:  <clips-dir>/strips/<clip>.npy (from 30f_make_v2n_strips.py / 31a_make_guifir_clips.py)
Variants (``v2n.transform_strips``): --flip none|h|v|hv, --layout mid|top, --norm (piano brightness/contrast).
System name: v2n_zeroshot for the default (none, mid, no norm), else v2n_zeroshot_<flip>_<layout>[_norm]
(e.g. v2n_zeroshot_v_top_norm), unless --system is given.
Writes: <clips-dir>/transcriptions/<system>/<clip>.mid   (scored by 30e like any system)
        <clips-dir>/v2n_probs[/<system>]/<clip>.npz      (stitched onset/offset/frame probs; default system
                                                         directly in v2n_probs/, as in CP6)
        <clips-dir>/plots/v2n/<clip>[__<system>].png     (reference vs V2N roll, --plots clips)
Reference for the plots: <clips-dir>/midi (GAPS) or transcriptions/real_audio (pseudo-GT).
"""

import argparse
import time
from pathlib import Path

import numpy as np

from guitar2midi.baselines import v2n

CLIPS = Path("data/interim/baseline_clips")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=CLIPS)
    p.add_argument("--strips", type=Path, help="default: <clips-dir>/strips")
    p.add_argument("--checkpoint", type=Path, default=v2n.V2N_CKPT)
    p.add_argument("--flip", default="none", choices=v2n.FLIPS)
    p.add_argument("--layout", default="mid", choices=v2n.LAYOUTS)
    p.add_argument("--norm", action="store_true", help="match brightness/contrast to V2N's piano strips")
    p.add_argument("--system", help="output name (default from the variant)")
    p.add_argument("--limit", type=int, help="only the first N clips (testing)")
    p.add_argument("--plots", nargs="*", default=None,
                   help="clips to plot (default: 3 spread over the clip list; missing ones are skipped)")
    args = p.parse_args()

    clips_dir = args.clips_dir
    strips_dir = args.strips or clips_dir / "strips"
    default = args.flip == "none" and args.layout == "mid" and not args.norm
    system = args.system or ("v2n_zeroshot" if default else
                             f"v2n_zeroshot_{args.flip}_{args.layout}" + ("_norm" if args.norm else ""))
    clips = sorted(f.stem for f in strips_dir.glob("*.npy") if not f.stem.endswith(("_valid", "_corners")))
    if args.limit:
        clips = clips[:args.limit]
    plots = args.plots if args.plots is not None else clips[:: max(len(clips) // 3, 1)][:3]
    if clips_dir == CLIPS and args.plots is None:
        plots = ["019_Vpswc_c1", "212_y41wc_c2", "291_3Sswc_c3"]  # the CP6 plot clips
    ref_dir, ref_label = ((clips_dir / "midi", "ground truth (GAPS MIDI)") if (clips_dir / "midi").is_dir() else
                          (clips_dir / "transcriptions" / "real_audio", "reference (transcription of the real audio)"))
    out_midi = clips_dir / "transcriptions" / system
    out_probs = clips_dir / "v2n_probs" if system == "v2n_zeroshot" else clips_dir / "v2n_probs" / system
    out_probs.mkdir(parents=True, exist_ok=True)

    model = v2n.V2NTranscriber(args.checkpoint)
    print(f"V2N {args.checkpoint.name} on {model.device}, repo {v2n.V2N_REPO}; {len(clips)} clips; "
          f"variant flip={args.flip} layout={args.layout} -> {system}")
    t0 = time.time()
    n_notes, max_on = [], []
    for clip in clips:
        strips = v2n.transform_strips(np.load(strips_dir / f"{clip}.npy"), args.flip, args.layout, args.norm)
        probs = model.probs(strips)
        notes = model.decode(probs)
        v2n.write_midi(notes, out_midi / f"{clip}.mid")
        np.savez_compressed(out_probs / f"{clip}.npz", **{k: v.astype(np.float16) for k, v in probs.items()})
        n_notes.append(len(notes))
        max_on.append(float(probs["onset_probs"].max()))
        if clip in plots:
            valid_f = strips_dir / f"{clip}_valid.npy"
            suffix = "" if system == "v2n_zeroshot" else f"__{system}"
            v2n.plot_roll(strips, probs, notes, ref_dir / f"{clip}.mid",
                          np.load(valid_f) if valid_f.exists() else None,
                          clips_dir / "plots" / "v2n" / f"{clip}{suffix}.png", title=f"{clip}: {system}",
                          ref_label=ref_label)
        print(f"{clip}: {len(notes):3d} notes, max onset prob {max_on[-1]:.2f}, "
              f"frames>0.5 {(probs['frame_probs'] > 0.5).mean():.3%}", flush=True)
    print(f"done in {time.time() - t0:.0f} s; notes per clip: median {np.median(n_notes):.0f}, total {sum(n_notes)}, "
          f"zero-note clips {sum(n == 0 for n in n_notes)}/{len(n_notes)}, max onset prob median "
          f"{np.median(max_on):.3f} / max {max(max_on):.3f} -> {out_midi}")


if __name__ == "__main__":
    main()
