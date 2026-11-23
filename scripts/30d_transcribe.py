"""Audio -> MIDI for the baseline study (CP3: real audio = ceiling; CP7: generated audio).

Runs in the ``amt`` env (conda env create -f envs/amt.yml; then pip install -e . --no-deps):

    conda activate amt
    python scripts/30d_transcribe.py --system real_audio                       # data/interim/baseline_clips/audio
    python scripts/30d_transcribe.py --system mmaudio_noprompt --audio-dir <dir with generated wavs>

Writes data/interim/baseline_clips/transcriptions/<system>/<clip>.mid
"""

import argparse
from pathlib import Path

from guitar2midi.baselines.transcribe import CHECKPOINTS, transcribe_dir

CLIPS = Path("data/interim/baseline_clips")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True, help="name of the audio source, used as output folder")
    p.add_argument("--audio-dir", type=Path, default=CLIPS / "audio")
    p.add_argument("--checkpoint", default="gaps_paper",
                   help=f"one of {list(CHECKPOINTS)} or a file name in the HF repo. Default gaps_paper: same "
                        "onset F1 as 'gaps' on our clips (93.5), better frame F1 (83.0 vs 80.8); used for all systems")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = CLIPS / "transcriptions" / args.system
    written = transcribe_dir(args.audio_dir, out, args.checkpoint, overwrite=args.overwrite)
    print(f"{len(written)} new MIDI files in {out}")


if __name__ == "__main__":
    main()
