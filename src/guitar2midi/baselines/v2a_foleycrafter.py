"""Video-to-audio with FoleyCrafter (Zhang et al., IJCV 2026). Runs in the ``foleycrafter`` env (envs/foleycrafter.yml).

FoleyCrafter = the Auffusion text-to-audio diffusion model (mel spectrogram, 16 kHz, 10.24 s canvas) plus two
video adapters: a *semantic* adapter (CLIP image embeddings of up to 150 frames, averaged into one vector, so it
says *what* sounds, not when) and a *temporal* adapter (a video onset detector, ``VideoOnsetNet``, on 112x112
centre crops, whose on/off sequence steers *when* sound happens). Text prompt optional.

The repo's ``inference.py`` is called unchanged (cwd = repo; weights are downloaded there to ``checkpoints/`` and
to the HF cache on first use): 25 PNDM steps, semantic scale 1.0, temporal scale 0.2, seed set once per call
(clips processed in sorted order, so a rerun on the same folder reproduces the outputs).
Output is cut to the clip duration.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(os.environ.get("FOLEYCRAFTER_REPO", Path.home() / "external" / "FoleyCrafter"))


def generate(videos: dict[str, Path], out_dir: Path, prompt: str = "", seed: int = 42,
             overwrite: bool = False) -> list[Path]:
    """Generate ``out_dir/<clip>.wav`` (16 kHz) for every clip id -> mp4 in ``videos``. Returns the new files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = {c: v for c, v in videos.items() if overwrite or not (out_dir / f"{c}.wav").exists()}
    if not todo:
        return []
    with tempfile.TemporaryDirectory() as tmp, open(out_dir / "foleycrafter.log", "a") as log:
        vids, res = Path(tmp) / "videos", Path(tmp) / "out"
        vids.mkdir()
        for c, v in todo.items():
            (vids / f"{c}.mp4").symlink_to(Path(v).resolve())
        cmd = [sys.executable, "inference.py", "--input", str(vids), "--save_dir", str(res), "--seed", str(seed),
               "--prompt", prompt]
        print("  $", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(r.stdout)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise RuntimeError(f"FoleyCrafter failed ({r.returncode})")
        written = []
        for c in todo:
            src = res / "audio" / f"{c.replace('+', ' ')}.wav"
            if not src.exists():
                raise RuntimeError(f"FoleyCrafter wrote no audio for {c}")
            shutil.move(str(src), out_dir / f"{c}.wav")
            written.append(out_dir / f"{c}.wav")
    return written
