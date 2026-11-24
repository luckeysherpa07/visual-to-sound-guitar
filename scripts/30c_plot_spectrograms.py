"""CP4 check: spectrograms of real vs generated audio for the listening-example clips (env guitar2midi).

    python scripts/30c_plot_spectrograms.py            # clips that have examples/<clip>__real.mp4
    python scripts/30c_plot_spectrograms.py --clips-dir data/interim/baseline_clips_magcil

Writes <clips-dir>/examples/<clip>__spectrograms.png (one row per audio source, reference notes drawn as white
lines: GT MIDI if <clips-dir>/midi exists (GAPS), else the real-audio transcription (pseudo-GT)).
"""

import argparse
import csv
from pathlib import Path

from guitar2midi.baselines.audio_plots import plot_spectrograms
from guitar2midi.evaluation.note_metrics import load_notes

CLIPS = Path("data/interim/baseline_clips")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=CLIPS)
    p.add_argument("--clips", nargs="+", help="default: all clips with listening examples")
    args = p.parse_args()
    clips_dir = args.clips_dir
    ref_dir = clips_dir / "midi" if (clips_dir / "midi").is_dir() else clips_dir / "transcriptions" / "real_audio"
    with open(clips_dir / "clips.csv") as f:
        lengths = {r["clip_id"]: float(r.get("length_s") or 8.0) for r in csv.DictReader(f)}

    clips = args.clips or sorted(f.name.split("__")[0] for f in (clips_dir / "examples").glob("*__real.mp4"))
    systems = sorted(d.name for d in (clips_dir / "generated").iterdir() if d.is_dir())
    for c in clips:
        audio = {"real audio": clips_dir / "audio" / f"{c}.wav"}
        audio.update({s: clips_dir / "generated" / s / f"{c}.wav" for s in systems
                      if (clips_dir / "generated" / s / f"{c}.wav").exists()})
        dst = clips_dir / "examples" / f"{c}__spectrograms.png"
        plot_spectrograms(audio, dst, c, gt_notes=load_notes(ref_dir / f"{c}.mid"), duration=lengths.get(c, 8.0),
                          ref_label="GT notes" if ref_dir.name == "midi" else "reference notes (real-audio transcription)")
        print(dst)


if __name__ == "__main__":
    main()
