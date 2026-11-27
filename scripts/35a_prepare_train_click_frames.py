"""CP17 step 17a: shots of the GAPS train/val videos and one frame per chosen shot for corner clicking.

    python scripts/35a_prepare_train_click_frames.py            # all train + val pieces of the CP16 split
    python scripts/35a_prepare_train_click_frames.py --limit 5  # quick test

Pieces = train + val rows of data/processed/mmaudio_lora/split.csv (CP16 performer split; test pieces never).
Per video: hard cuts at 25 fps (same rule as CP5, streamed: preprocessing.video.detect_cuts_file), then the
longest shots are chosen until >= --coverage of the video is covered, at most --max-shots, each >= --min-shot-s
(V2N records need >= 2 s). The middle frame of each chosen shot is written at full resolution:
    data/interim/v2n_train/click_frames/<piece>__s<k>/f<frame>.png   (frame = 25 fps index in the whole video)
    data/interim/v2n_train/shots.json   per piece: n_frames, cuts, chosen shots [start, end) + click frame
    data/interim/v2n_train/shots.csv    per piece: n shots, chosen, coverage

Then click (GAPS *train* split: may later become FPE training data):
    python scripts/01b_annotate_corners.py --image-glob "data/interim/v2n_train/click_frames/*/*.png" \\
        --source gaps_train --no-shuffle --out data/labels/fretboard_corners/gaps_train.jsonl
(--web for the browser version; p pastes the previous shot's clicks (same camera angle), s skips a shot
without a usable fretboard view)
"""

import argparse
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from guitar2midi.preprocessing.video import detect_cuts_file, shots

SPLIT = Path("data/processed/mmaudio_lora/split.csv")
VIDEOS = Path("data/raw/gaps/videos")
OUT = Path("data/interim/v2n_train")
FPS = 25.0


def choose_shots(n: int, cuts: list[int], coverage: float, max_shots: int, min_len: int) -> list[tuple[int, int]]:
    """Longest shots first until ``coverage`` of the n frames is covered (at most ``max_shots``), in time order."""
    cand = sorted((s for s in shots(n, cuts) if s[1] - s[0] >= min_len), key=lambda s: s[0] - s[1])
    chosen, covered = [], 0
    for s in cand:
        if len(chosen) >= max_shots or covered >= coverage * n:
            break
        chosen.append(s)
        covered += s[1] - s[0]
    return sorted(chosen)


def process(piece: str, args) -> dict:
    video = VIDEOS / f"{piece}.mp4"
    n, cuts = detect_cuts_file(video, FPS)
    chosen = choose_shots(n, cuts, args.coverage, args.max_shots, int(args.min_shot_s * FPS))
    out = []
    for k, (a, b) in enumerate(chosen):
        f = (a + b) // 2
        d = OUT / "click_frames" / f"{piece}__s{k}"
        d.mkdir(parents=True, exist_ok=True)
        png = d / f"f{f:06d}.png"
        if not png.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{f / FPS:.3f}", "-i", str(video),
                            "-frames:v", "1", str(png)], check=True)
        out.append({"shot": k, "start": a, "end": b, "click_frame": f})
    return {"piece": piece, "n_frames": n, "cuts": cuts, "chosen": out}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coverage", type=float, default=0.7)
    p.add_argument("--max-shots", type=int, default=5)
    p.add_argument("--min-shot-s", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    with open(SPLIT) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
    pieces = [r["piece"] for r in rows if (VIDEOS / f"{r['piece']}.mp4").exists()]
    missing = [r["piece"] for r in rows if not (VIDEOS / f"{r['piece']}.mp4").exists()]
    if missing:
        print(f"no video for {len(missing)} pieces: {missing}")
    pieces = pieces[: args.limit]
    split_of = {r["piece"]: r["split"] for r in rows}

    OUT.mkdir(parents=True, exist_ok=True)
    info = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(process, pieces, [args] * len(pieces))):
            info[res["piece"]] = res
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(pieces)}", flush=True)
    (OUT / "shots.json").write_text(json.dumps(info, indent=1))

    with open(OUT / "shots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["piece", "split", "minutes", "n_cuts", "n_shots", "n_chosen", "coverage"])
        for pc, r in info.items():
            n = r["n_frames"]
            cov = sum(s["end"] - s["start"] for s in r["chosen"]) / n
            w.writerow([pc, split_of[pc], f"{n / FPS / 60:.2f}", len(r["cuts"]), len(r["cuts"]) + 1,
                        len(r["chosen"]), f"{cov:.3f}"])

    n_cut = [len(r["cuts"]) for r in info.values()]
    covs = [sum(s["end"] - s["start"] for s in r["chosen"]) / r["n_frames"] for r in info.values()]
    tot = sum(r["n_frames"] for r in info.values()) / FPS / 3600
    kept = sum(c * r["n_frames"] for c, r in zip(covs, info.values())) / FPS / 3600
    clicks = sum(len(r["chosen"]) for r in info.values())
    print(f"\n{len(info)} videos, {tot:.2f} h; single-shot videos: {sum(c == 0 for c in n_cut)}; "
          f"cuts per video median {sorted(n_cut)[len(n_cut) // 2]}, max {max(n_cut)}")
    print(f"chosen shots to click: {clicks}; covered {kept:.2f} h ({kept / tot:.0%}); "
          f"videos below 70 % coverage: {sum(c < args.coverage for c in covs)}")


if __name__ == "__main__":
    main()
