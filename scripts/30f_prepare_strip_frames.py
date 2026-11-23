"""CP5 step 1: detect camera cuts per baseline clip and export one frame per clip for corner clicking.

    python scripts/30f_prepare_strip_frames.py

The frame to click is the middle of the longest shot (between hard cuts). Writes
    data/interim/baseline_clips/strip_frames/<clip_id>/f<frame>.png   (full resolution, 25 fps index)
    data/interim/baseline_clips/shots.json                            cuts, longest shot, reference frame

Then click (GAPS *test* clips: these labels must never be used to train the fretboard estimator):
    python scripts/01b_annotate_corners.py --image-glob "data/interim/baseline_clips/strip_frames/*/*.png" \\
        --source gaps_test_baseline --no-shuffle --out data/labels/fretboard_corners/gaps_test_baseline_EVAL_ONLY.jsonl
(add --web for the browser version; press p to paste the previous clip's corners)
"""

import csv
import json
from pathlib import Path

import cv2

from guitar2midi.preprocessing.video import detect_cuts, longest_shot, read_frames, shots

CLIPS = Path("data/interim/baseline_clips")


def main() -> None:
    with open(CLIPS / "clips.csv") as f:
        clips = [r["clip_id"] for r in csv.DictReader(f) if r["keep"] == "1"]
    out_dir = CLIPS / "strip_frames"
    info = {}
    for cid in clips:
        frames = read_frames(CLIPS / "video" / f"{cid}.mp4")
        cuts = detect_cuts(frames)
        a, b = longest_shot(len(frames), cuts)
        ref = (a + b) // 2
        d = out_dir / cid
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
        cv2.imwrite(str(d / f"f{ref:03d}.png"), frames[ref])
        info[cid] = {"n_frames": len(frames), "cuts": cuts, "shots": shots(len(frames), cuts),
                     "shot": [a, b], "ref_frame": ref}
        if cuts:
            print(f"{cid}: cuts at frames {cuts} -> clicking frame {ref} of shot {a}-{b} "
                  f"({(b - a) / len(frames):.0%} of the clip)")
    (CLIPS / "shots.json").write_text(json.dumps(info, indent=1))
    with_cuts = [c for c in info if info[c]["cuts"]]
    print(f"\n{len(info)} clips, {len(with_cuts)} with hard cuts; frames in {out_dir}")


if __name__ == "__main__":
    main()
