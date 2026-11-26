"""CP16 step 16a: MMAudio features of the LoRA training windows (env mmaudio).

    PYTHONNOUSERSITE=1 python scripts/34b_extract_lora_features.py              # all kept windows (resumes)
    PYTHONNOUSERSITE=1 python scripts/34b_extract_lora_features.py --limit 1    # one video, for a check

Input:  data/processed/mmaudio_lora/windows.csv (34a), GAPS train videos (frames + their own audio track)
Writes: data/processed/mmaudio_lora/features/<piece>.npz   per video: start_s (N,), clip (N,64,1024) fp16,
                                                           sync (N,192,768) fp16, mean/std (N,345,40) fp32, rms_db (N,)
        data/processed/mmaudio_lora/text_features.npy      prompt "solo classical guitar" (77, 1024)
See ``guitar2midi.baselines.mmaudio_lora`` for how the features match MMAudio's generation and training.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from guitar2midi.baselines import mmaudio_lora as L

GAPS = Path("data/raw/gaps")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=Path("data/processed/mmaudio_lora"))
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    by_piece = defaultdict(list)
    for r in L.read_windows(args.dir / "windows.csv"):
        if r["keep"] == "1":
            by_piece[r["piece"]].append(float(r["start_s"]))
    pieces = sorted(by_piece)[:args.limit]
    out = args.dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    ex = L.WindowFeatureExtractor()
    np.save(args.dir / "text_features.npy", ex.text(L.PROMPT))
    t0, done = time.time(), 0
    for k, piece in enumerate(pieces):
        dst = out / f"{piece}.npz"
        if dst.exists():
            continue
        t1 = time.time()
        res = ex.video_windows(GAPS / "videos" / f"{piece}.mp4", by_piece[piece], args.dir / "tmp")
        starts = sorted(res)
        if not starts:
            print(f"{piece}: no windows", flush=True)
            continue
        np.savez(dst, start_s=np.array(starts),
                 clip=np.stack([res[s]["clip"] for s in starts]).astype(np.float16),
                 sync=np.stack([res[s]["sync"] for s in starts]).astype(np.float16),
                 mean=np.stack([res[s]["mean"] for s in starts]), std=np.stack([res[s]["std"] for s in starts]),
                 rms_db=np.array([res[s]["rms_db"] for s in starts]))
        done += 1
        print(f"[{k + 1}/{len(pieces)}] {piece}: {len(starts)}/{len(by_piece[piece])} windows in "
              f"{time.time() - t1:.0f} s, rms {np.median([res[s]['rms_db'] for s in starts]):.0f} dB", flush=True)
    print(f"done: {done} videos in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
