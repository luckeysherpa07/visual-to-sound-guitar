"""CP4: generate audio from the silent baseline clips with MMAudio (env ``mmaudio``).

    conda activate mmaudio
    python scripts/30c_run_mmaudio.py                       # both systems below, all clips
    python scripts/30c_run_mmaudio.py --limit 1             # quick test on one clip
    python scripts/30c_run_mmaudio.py --clips-dir data/interim/baseline_clips_guifir   # other clip sets (CP9)
    python scripts/30c_run_mmaudio.py --clips-dir data/interim/baseline_clips_magcil

Systems (same seed 42 for every clip):
    mmaudio_noprompt   video only
    mmaudio_prompt     video + text prompt; default per clip set (PROMPTS): "solo classical guitar" (GAPS),
                       "solo acoustic guitar" (GuiFiR-Vid), "solo electric guitar" (magcil); --prompt overrides.
                       The prompt used is written to generated/mmaudio_prompt/prompt.txt.

Clips are generated at their own length (MMAudio reads the video duration; magcil bend clips are ~3.3 s).
Writes <clips-dir>/generated/<system>/<clip>.wav and, for listening, <clips-dir>/examples/<clip>__{real,<system>}.mp4.
Then transcribe in env amt:
    python scripts/30d_transcribe.py --clips-dir <clips-dir> --system <system> --audio-dir <clips-dir>/generated/<system>
"""

import argparse
import csv
import shutil
import tempfile
import time
from pathlib import Path

from guitar2midi.baselines.v2a_mmaudio import MMAudioGenerator, mux

CLIPS = Path("data/interim/baseline_clips")
SYSTEMS = ("mmaudio_noprompt", "mmaudio_prompt")
PROMPTS = {"baseline_clips": "solo classical guitar", "baseline_clips_guifir": "solo acoustic guitar",
           "baseline_clips_magcil": "solo electric guitar"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=CLIPS)
    p.add_argument("--prompt", help="text prompt of mmaudio_prompt (default: PROMPTS by clip set)")
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    p.add_argument("--variant", default="large_44k_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="only the first N clips (testing)")
    p.add_argument("--examples", type=int, default=5, help="clips muxed into mp4 for listening")
    args = p.parse_args()
    clips_dir = args.clips_dir
    prompts = {"mmaudio_noprompt": "", "mmaudio_prompt": args.prompt or PROMPTS[clips_dir.name]}

    with open(clips_dir / "clips.csv") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f) if r["keep"] == "1"]
    if args.limit:
        clips = clips[: args.limit]

    gen = MMAudioGenerator(args.variant)
    with tempfile.TemporaryDirectory() as tmp:  # generate_dir works on a folder: link the selected clips
        for c in clips:
            (Path(tmp) / f"{c}.mp4").symlink_to((clips_dir / "video" / f"{c}.mp4").resolve())
        for system in args.systems:
            t0 = time.time()
            out = clips_dir / "generated" / system
            print(f"{system}: prompt={prompts[system]!r} -> {out}")
            written = gen.generate_dir(Path(tmp), out, prompt=prompts[system], seed=args.seed)
            (out / "prompt.txt").write_text(prompts[system] + "\n")
            print(f"{system}: {len(written)} new clips in {time.time() - t0:.0f} s")

    ex_dir = clips_dir / "examples"
    for c in clips[:: max(len(clips) // max(args.examples, 1), 1)][: args.examples]:
        mux(clips_dir / "video" / f"{c}.mp4", clips_dir / "audio" / f"{c}.wav", ex_dir / f"{c}__real.mp4")
        for system in args.systems:
            mux(clips_dir / "video" / f"{c}.mp4", clips_dir / "generated" / system / f"{c}.wav", ex_dir / f"{c}__{system}.mp4")
    print(f"listening examples: {ex_dir}")


if __name__ == "__main__":
    main()
