"""CP14: Audeo zero-shot (released piano weights) on the guitar fretboard strips (env guitar2midi).

    python scripts/32b_run_audeo.py                                          # GAPS, all clips with strips
    python scripts/32b_run_audeo.py --clips-dir data/interim/baseline_clips_guifir
    python scripts/32b_run_audeo.py --flip v                                 # variant: strip flipped top <-> bottom
    python scripts/32b_run_audeo.py --strips data/interim/baseline_clips/strips_full --suffix full

Input:  <clips-dir>/strips/<clip>.npy (T, 144, 800) uint8, 25 fps, nut left (same strips as V2N, CP5/CP7/CP8)
Model:  ``guitar2midi.baselines.audeo`` (stage 1 Video2RollNet -> stage 2 Roll2MidiNet), checked against Audeo's
        own outputs by 32a_check_audeo.py.
Writes two systems (scored by 30e like any other):
        <clips-dir>/transcriptions/audeo_zeroshot[_<variant>]/<clip>.mid   full Audeo (Roll2Midi output)
        <clips-dir>/transcriptions/audeo_roll[_<variant>]/<clip>.mid       stage 1 only (roll, sigmoid >= 0.4)
        <clips-dir>/audeo_probs/<system>/<clip>.npz                         stage-1 logits (float16)
        <clips-dir>/plots/audeo/<clip>[__<variant>].png                     reference vs Audeo roll (3 clips)
Variant name: <flip> and/or --suffix, e.g. audeo_zeroshot_v, audeo_zeroshot_full.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from guitar2midi.baselines import audeo

CLIPS = Path("data/interim/baseline_clips")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=CLIPS)
    p.add_argument("--strips", type=Path, help="default: <clips-dir>/strips")
    p.add_argument("--flip", default="none", choices=["none", "h", "v", "hv"])
    p.add_argument("--suffix", default="", help="extra variant name, e.g. full for strips_full")
    p.add_argument("--limit", type=int, help="only the first N clips (testing)")
    p.add_argument("--plots", nargs="*", default=None, help="clips to plot (default: 3 spread over the list)")
    args = p.parse_args()

    clips_dir = args.clips_dir
    strips_dir = args.strips or clips_dir / "strips"
    variant = "_".join(v for v in ([args.flip] if args.flip != "none" else []) + ([args.suffix] if args.suffix else []))
    sys_full = "audeo_zeroshot" + (f"_{variant}" if variant else "")
    sys_roll = "audeo_roll" + (f"_{variant}" if variant else "")
    clips = sorted(f.stem for f in strips_dir.glob("*.npy") if not f.stem.endswith(("_valid", "_corners")))
    if args.limit:
        clips = clips[:args.limit]
    plots = args.plots if args.plots is not None else clips[:: max(len(clips) // 3, 1)][:3]
    if clips_dir == CLIPS and args.plots is None:
        plots = ["019_Vpswc_c1", "212_y41wc_c2", "291_3Sswc_c3"]  # the CP6 plot clips
    ref_dir, ref_label = ((clips_dir / "midi", "ground truth (GAPS MIDI)") if (clips_dir / "midi").is_dir() else
                          (clips_dir / "transcriptions" / "real_audio", "reference (transcription of the real audio)"))
    valid_dir = clips_dir / "strips"  # masks always come from the fretboard strips (same frames for strips_full)

    model = audeo.AudeoTranscriber()
    print(f"Audeo on {model.device}, weights {audeo.AUDEO_DATA}; {len(clips)} clips from {strips_dir}; "
          f"flip={args.flip} -> {sys_full}, {sys_roll}")
    t0 = time.time()
    stats = {sys_full: [], sys_roll: []}
    max_p = []
    for clip in clips:
        s = np.load(strips_dir / f"{clip}.npy")
        if "h" in args.flip:
            s = s[:, :, ::-1]
        if "v" in args.flip:
            s = s[:, ::-1, :]
        s = np.ascontiguousarray(s)
        out = model.transcribe(s)
        notes_full, notes_roll = audeo.roll_to_notes(out["midi"]), audeo.roll_to_notes(out["roll"])
        audeo.write_midi(notes_full, clips_dir / "transcriptions" / sys_full / f"{clip}.mid")
        audeo.write_midi(notes_roll, clips_dir / "transcriptions" / sys_roll / f"{clip}.mid")
        pd = clips_dir / "audeo_probs" / sys_full
        pd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pd / f"{clip}.npz", logits=out["logits"].astype(np.float16))
        stats[sys_full].append(len(notes_full))
        stats[sys_roll].append(len(notes_roll))
        max_p.append(float(1 / (1 + np.exp(-out["logits"].max()))))
        if clip in plots:
            vf = valid_dir / f"{clip}_valid.npy"
            audeo.plot_roll(s, out, notes_full, ref_dir / f"{clip}.mid", np.load(vf) if vf.exists() else None,
                            clips_dir / "plots" / "audeo" / f"{clip}{'__' + variant if variant else ''}.png",
                            title=f"{clip}: {sys_full}", ref_label=ref_label)
        print(f"{clip}: {len(notes_full):3d} notes (roll {len(notes_roll):3d}), max key prob {max_p[-1]:.2f}, "
              f"roll frames on {out['roll'].mean():.2%}", flush=True)
    for name, n in stats.items():
        print(f"{name}: notes per clip median {np.median(n):.0f}, total {sum(n)}, zero-note clips "
              f"{sum(x == 0 for x in n)}/{len(n)}")
    print(f"max key prob per clip: median {np.median(max_p):.3f}, max {max(max_p):.3f}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
