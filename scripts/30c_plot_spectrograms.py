"""CP4 check: spectrograms of real vs generated audio for the listening-example clips (env guitar2midi).

    python scripts/30c_plot_spectrograms.py            # clips that have examples/<clip>__real.mp4

Writes data/interim/baseline_clips/examples/<clip>__spectrograms.png (one row per audio source,
GT notes drawn as white lines).
"""

import argparse
from pathlib import Path

from guitar2midi.baselines.audio_plots import plot_spectrograms
from guitar2midi.evaluation.note_metrics import load_notes

CLIPS = Path("data/interim/baseline_clips")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", nargs="+", help="default: all clips with listening examples")
    args = p.parse_args()

    clips = args.clips or sorted(f.name.split("__")[0] for f in (CLIPS / "examples").glob("*__real.mp4"))
    systems = sorted(d.name for d in (CLIPS / "generated").iterdir() if d.is_dir())
    for c in clips:
        audio = {"real audio": CLIPS / "audio" / f"{c}.wav"}
        audio.update({s: CLIPS / "generated" / s / f"{c}.wav" for s in systems
                      if (CLIPS / "generated" / s / f"{c}.wav").exists()})
        dst = CLIPS / "examples" / f"{c}__spectrograms.png"
        plot_spectrograms(audio, dst, c, gt_notes=load_notes(CLIPS / "midi" / f"{c}.mid"))
        print(dst)


if __name__ == "__main__":
    main()
