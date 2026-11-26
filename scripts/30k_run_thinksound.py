"""CP13: generate audio from the silent baseline clips with ThinkSound (env ``thinksound``).

    conda activate thinksound
    python scripts/30k_run_thinksound.py --limit 1                                   # quick test
    python scripts/30k_run_thinksound.py                                             # GAPS
    python scripts/30k_run_thinksound.py --clips-dir data/interim/baseline_clips_guifir
    python scripts/30k_run_thinksound.py --clips-dir data/interim/baseline_clips_magcil

Systems (seed 42), same prompts as MMAudio (30c PROMPTS) so the rows are comparable:
    thinksound_noprompt   video only (empty caption and chain-of-thought text)
    thinksound_prompt     video + caption ("solo classical / acoustic / electric guitar") + a fixed
                          chain-of-thought text built from it (``v2a_thinksound.cot_text``)
Writes <clips-dir>/generated/<system>/<clip>.wav (44.1 kHz; whole seconds, see v2a_thinksound) + prompt.txt,
and listening mp4s for the same example clips as MMAudio (<clips-dir>/examples/<clip>__<system>.mp4).
Then transcribe (env amt) and evaluate as for MMAudio:
    python scripts/30d_transcribe.py --clips-dir <clips-dir> --system <system> --audio-dir <clips-dir>/generated/<system>
"""

import argparse
import csv
import importlib.util
import time
from pathlib import Path

from guitar2midi.baselines import v2a_thinksound as TS
from guitar2midi.baselines.v2a_mmaudio import mux

CLIPS = Path("data/interim/baseline_clips")
SYSTEMS = ("thinksound_noprompt", "thinksound_prompt")


def mmaudio_prompts() -> dict:
    spec = importlib.util.spec_from_file_location("run_mmaudio", Path(__file__).with_name("30c_run_mmaudio.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROMPTS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=CLIPS)
    p.add_argument("--prompt", help="caption of thinksound_prompt (default: MMAudio's prompt for the clip set)")
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full-precision", action="store_true", help="no --use_half in feature extraction")
    p.add_argument("--limit", type=int, help="only the first N clips (testing)")
    args = p.parse_args()
    cd = args.clips_dir
    prompts = {"thinksound_noprompt": "", "thinksound_prompt": args.prompt or mmaudio_prompts()[cd.name]}

    with open(cd / "clips.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["keep"] == "1"]
    if args.limit:
        rows = rows[: args.limit]
    videos = {r["clip_id"]: cd / "video" / f"{r['clip_id']}.mp4" for r in rows}
    durations = {r["clip_id"]: float(r.get("length_s") or 8.0) for r in rows}

    for system in args.systems:
        t0 = time.time()
        out = cd / "generated" / system
        written = TS.generate(videos, durations, out, prompt=prompts[system], seed=args.seed,
                              use_half=not args.full_precision)
        (out / "prompt.txt").write_text(f"caption: {prompts[system]}\ncaption_cot: {TS.cot_text(prompts[system])}\n")
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
