"""CP4: generate audio from the silent baseline clips with MMAudio (env ``mmaudio``).

    conda activate mmaudio
    python scripts/30c_run_mmaudio.py                       # both systems below, all clips
    python scripts/30c_run_mmaudio.py --limit 1             # quick test on one clip

Systems (same seed 42 for every clip):
    mmaudio_noprompt   video only
    mmaudio_prompt     video + text prompt "solo classical guitar"

Writes data/interim/baseline_clips/generated/<system>/<clip>.wav and, for listening,
data/interim/baseline_clips/examples/<clip>__{real,<system>}.mp4 for a few clips.
Then transcribe in env amt:  python scripts/30d_transcribe.py --system <system> --audio-dir .../generated/<system>
"""

import argparse
import csv
import shutil
import tempfile
import time
from pathlib import Path

from guitar2midi.baselines.v2a_mmaudio import MMAudioGenerator, mux

CLIPS = Path("data/interim/baseline_clips")
SYSTEMS = {"mmaudio_noprompt": "", "mmaudio_prompt": "solo classical guitar"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    p.add_argument("--variant", default="large_44k_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="only the first N clips (testing)")
    p.add_argument("--examples", type=int, default=5, help="clips muxed into mp4 for listening")
    args = p.parse_args()

    with open(CLIPS / "clips.csv") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f) if r["keep"] == "1"]
    if args.limit:
        clips = clips[: args.limit]

    gen = MMAudioGenerator(args.variant)
    with tempfile.TemporaryDirectory() as tmp:  # generate_dir works on a folder: link the selected clips
        for c in clips:
            (Path(tmp) / f"{c}.mp4").symlink_to((CLIPS / "video" / f"{c}.mp4").resolve())
        for system in args.systems:
            t0 = time.time()
            out = CLIPS / "generated" / system
            print(f"{system}: prompt={SYSTEMS[system]!r} -> {out}")
            written = gen.generate_dir(Path(tmp), out, prompt=SYSTEMS[system], seed=args.seed)
            print(f"{system}: {len(written)} new clips in {time.time() - t0:.0f} s")

    ex_dir = CLIPS / "examples"
    for c in clips[:: max(len(clips) // max(args.examples, 1), 1)][: args.examples]:
        mux(CLIPS / "video" / f"{c}.mp4", CLIPS / "audio" / f"{c}.wav", ex_dir / f"{c}__real.mp4")
        for system in args.systems:
            mux(CLIPS / "video" / f"{c}.mp4", CLIPS / "generated" / system / f"{c}.wav", ex_dir / f"{c}__{system}.mp4")
    print(f"listening examples: {ex_dir}")


if __name__ == "__main__":
    main()
