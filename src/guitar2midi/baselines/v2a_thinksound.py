"""Video-to-audio with ThinkSound (Liu et al., NeurIPS 2025). Runs in the ``thinksound`` env (envs/thinksound.yml).

ThinkSound conditions on the video (MetaCLIP at 8 fps on 224x224 squashed frames; Synchformer at 25 fps on
224x224 centre crops) and on two texts: a short
``caption`` (MetaCLIP text encoder) and a longer chain-of-thought ``caption_cot`` (T5-v1.1-XL). In the paper the
CoT comes from a multimodal LLM; at inference it is just a text field of the input CSV, so we pass a fixed text
(or nothing). No LLM runs here.

The repo's own two steps are called unchanged, with the repo as working directory (its paths are relative):
    extract_latents.py  video + texts -> features (.npz)        (``--use_half`` for 16 GB)
    eval_batch.py       features -> 44.1 kHz audio (24 Euler steps, CFG 5, ckpts/thinksound_light.ckpt)
Both steps need all videos of one call to have the same duration, and use ``int(duration)`` seconds, so clips are
processed in groups of equal whole-second length; a 7.33 s clip gives 7 s of audio (the rest scores as silence).
The output is peak-normalised by ThinkSound.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(os.environ.get("THINKSOUND_REPO", Path.home() / "external" / "ThinkSound"))


def cot_text(prompt: str) -> str:
    """Fixed chain-of-thought text for a guitar prompt ('' -> '')."""
    if not prompt:
        return ""
    return (f"Generate the sound of {prompt} played by the musician in the video. Each pluck or strum of the "
            f"strings visible in the video produces a note, with the pitch given by the fretting hand. Keep only "
            f"the guitar; no voices, no other instruments, no background noise.")


def _run(cmd: list[str], log) -> None:
    print("  $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write(r.stdout)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise RuntimeError(f"ThinkSound step failed ({r.returncode}): {cmd[1]}")


def generate(videos: dict[str, Path], durations: dict[str, float], out_dir: Path, prompt: str = "",
             seed: int = 42, use_half: bool = True, overwrite: bool = False) -> list[Path]:
    """Generate ``out_dir/<clip>.wav`` for every clip in ``videos`` (clip id -> mp4). Returns the new files."""
    import sys

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = {c: v for c, v in videos.items() if overwrite or not (out_dir / f"{c}.wav").exists()}
    groups: dict[int, list[str]] = defaultdict(list)
    for c in todo:
        groups[int(durations[c])].append(c)
    written = []
    py = sys.executable
    with tempfile.TemporaryDirectory() as tmp, open(out_dir / "thinksound.log", "a") as log:
        for dur, clips in sorted(groups.items()):
            g = Path(tmp) / f"d{dur}"
            vids, feats, res = g / "videos", g / "features", g / "audio"
            for d in (vids, feats, res):
                d.mkdir(parents=True)
            for c in clips:
                (vids / f"{c}.mp4").symlink_to(Path(todo[c]).resolve())
            with open(g / "prompts.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "caption", "caption_cot"])
                for c in clips:
                    # empty text -> ' ': ThinkSound reads the CSV with pandas, which turns '' into NaN
                    w.writerow([c, prompt or " ", cot_text(prompt) or " "])
            print(f"ThinkSound: {len(clips)} clips of {dur} s, prompt={prompt!r}", flush=True)
            cmd = [py, "extract_latents.py", "--root", str(vids), "--tsv_path", str(g / "prompts.csv"),
                   "--save-dir", str(feats), "--duration_sec", str(dur)]
            _run(cmd + (["--use_half"] if use_half else []), log)
            _run([py, "eval_batch.py", "--dataset-config", "ThinkSound/configs/multimodal_dataset_demo.json",
                  "--model-config", "ThinkSound/configs/model_configs/thinksound.json",
                  "--duration-sec", str(dur), "--results-dir", str(feats), "--save-dir", str(res),
                  "--seed", str(seed), "--num-workers", "4"], log)
            for c in clips:
                src = next(res.rglob(f"{c}.wav"), None)
                if src is None:
                    raise RuntimeError(f"ThinkSound wrote no audio for {c}")
                shutil.move(str(src), out_dir / f"{c}.wav")
                written.append(out_dir / f"{c}.wav")
    return written
