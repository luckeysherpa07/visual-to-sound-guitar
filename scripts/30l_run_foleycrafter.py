"""CP13: generate audio from the silent baseline clips with FoleyCrafter (env ``foleycrafter``).

    conda activate thinksound
    python scripts/30l_run_foleycrafter.py --limit 1                                   # quick test
    python scripts/30l_run_foleycrafter.py                                             # GAPS
    python scripts/30l_run_foleycrafter.py --clips-dir data/interim/baseline_clips_guifir
    python scripts/30l_run_foleycrafter.py --clips-dir data/interim/baseline_clips_magcil

Systems (seed 42), same prompts as MMAudio (30c PROMPTS) so the rows are comparable:
    foleycrafter_noprompt   video only (empty prompt)
    foleycrafter_prompt     video + text prompt ("solo classical / acoustic / electric guitar")
Writes <clips-dir>/generated/<system>/<clip>.wav (16 kHz, cut to the clip length; see v2a_foleycrafter) + prompt.txt,
and listening mp4s for the same example clips as MMAudio (<clips-dir>/examples/<clip>__<system>.mp4).
Then transcribe (env amt) and evaluate as for MMAudio:
    python scripts/30d_transcribe.py --clips-dir <clips-dir> --system <system> --audio-dir <clips-dir>/generated/<system>
"""

import argparse
import csv
import importlib.util
import time
from pathlib import Path

from guitar2midi.baselines import v2a_foleycrafter as FC
from guitar2midi.baselines.v2a_mmaudio import mux

CLIPS = Path("data/interim/baseline_clips")
SYSTEMS = ("foleycrafter_noprompt", "foleycrafter_prompt")


def mmaudio_prompts() -> dict:
    spec = importlib.util.spec_from_file_location("run_mmaudio", Path(__file__).with_name("30c_run_mmaudio.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROMPTS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=CLIPS)
    p.add_argument("--prompt", help="text prompt of foleycrafter_prompt (default: MMAudio's prompt for the clip set)")
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="only the first N clips (testing)")
    args = p.parse_args()
    cd = args.clips_dir
    prompts = {"foleycrafter_noprompt": "", "foleycrafter_prompt": args.prompt or mmaudio_prompts()[cd.name]}

    with open(cd / "clips.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["keep"] == "1"]
    if args.limit:
        rows = rows[: args.limit]
    videos = {r["clip_id"]: cd / "video" / f"{r['clip_id']}.mp4" for r in rows}

    for system in args.systems:
        t0 = time.time()
        out = cd / "generated" / system
        written = FC.generate(videos, out, prompt=prompts[system], seed=args.seed)
        (out / "prompt.txt").write_text(prompts[system] + "\n")
        print(f"{system}: {len(written)} new clips in {time.time() - t0:.0f} s -> {out}", flush=True)

    ex = cd / "examples"
    for real in sorted(ex.glob("*__real.mp4")):
        c = real.name[: -len("__real.mp4")]
        for system in args.systems:
            wav = cd / "generated" / system / f"{c}.wav"
            if c in videos and wav.exists():
                mux(videos[c], wav, ex / f"{c}__{system}.mp4")
    print(f"listening examples: {ex}")


if __name__ == "__main__":
    main()
