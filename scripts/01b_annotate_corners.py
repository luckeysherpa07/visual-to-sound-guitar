"""Click 4 fretboard corners per frame; the 126-point grid is generated from them.

Click order (physical, not image position):
    1. nut, high-E string     2. nut, low-E string
    3. 20th fret, high-E      4. 20th fret, low-E
If the 20th fret is hidden by the hand/body, click where it would be.

Keys:  enter/space = accept   u = undo click   r = reset
       s = skip (fretboard not usable, saved as visible=false)   q = quit
Progress is appended to the output file, so you can stop and resume.

Examples:
    # one random frame per GuiFiR-Vid train clip
    python scripts/01b_annotate_corners.py --guifir-split train --per-clip 1 \
        --out data/labels/fretboard_corners/guifir_train.jsonl
    # any folder of frames (e.g. GAPS train-split performers only!)
    python scripts/01b_annotate_corners.py --image-glob "data/raw/gaps/frames/**/*.png" \
        --source gaps --out data/labels/fretboard_corners/gaps.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from guitar2midi.data import guifir
from guitar2midi.preprocessing.fretboard_grid import corners_valid, draw_grid, grid_from_corners

DATA_ROOT = Path("data")
PROMPTS = ["1: nut / high-E", "2: nut / low-E", "3: fret 20 / high-E", "4: fret 20 / low-E"]
MAX_W, MAX_H = 1600, 900


def build_queue(args) -> list[dict]:
    rng = random.Random(args.seed)
    items = []
    if args.guifir_split:
        by_clip = defaultdict(list)
        for rec in guifir.load_split(args.guifir_split):
            by_clip[guifir.clip_id(rec)].append(rec)
        for cid in sorted(by_clip):
            for rec in rng.sample(by_clip[cid], min(args.per_clip, len(by_clip[cid]))):
                items.append(dict(
                    path=guifir.frame_path(rec, args.guifir_split),
                    source=f"guifir_{args.guifir_split}", clip_id=cid))
    else:
        for p in sorted(Path().glob(args.image_glob)):
            items.append(dict(path=p, source=args.source, clip_id=p.parent.name))
    rng.shuffle(items)
    return items[: args.limit] if args.limit else items


def annotate(img: np.ndarray, title: str) -> list | None | str:
    """Returns 4 corners, None (skip) or 'quit'."""
    scale = min(MAX_W / img.shape[1], MAX_H / img.shape[0], 1.0)
    pts: list[tuple[float, float]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        view = img.copy()
        valid = len(pts) == 4 and corners_valid(np.array(pts))
        if valid:
            view = draw_grid(view, grid_from_corners(np.array(pts)), thickness=2)
        for i, p in enumerate(pts):
            cv2.circle(view, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        view = cv2.resize(view, None, fx=scale, fy=scale)
        if len(pts) < 4:
            msg = PROMPTS[len(pts)]
        else:
            msg = "enter=accept  u=undo  r=reset" if valid else "INVALID (crossed) - u=undo r=reset"
        cv2.putText(view, msg + "   [s=skip q=quit]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            return "quit"
        if key == ord("s"):
            return None
        if key == ord("u") and pts:
            pts.pop()
        if key == ord("r"):
            pts.clear()
        if key in (13, 32) and valid:
            return [list(p) for p in pts]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--guifir-split", choices=["train"], help="sample frames from GuiFiR-Vid (train only: val/test are evaluation)")
    src.add_argument("--image-glob", help="glob of frame images")
    ap.add_argument("--source", default="images", help="source name for --image-glob")
    ap.add_argument("--per-clip", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["image"] for l in args.out.read_text().splitlines() if l.strip()}

    queue = [it for it in build_queue(args) if str(it["path"].relative_to(DATA_ROOT)) not in done]
    print(f"{len(done)} done, {len(queue)} to annotate")
    with open(args.out, "a") as f:
        for n, it in enumerate(queue):
            img = cv2.imread(str(it["path"]))
            if img is None:
                print("cannot read", it["path"])
                continue
            res = annotate(img, "corners")
            if res == "quit":
                break
            f.write(json.dumps(dict(
                image=str(it["path"].relative_to(DATA_ROOT)),
                source=it["source"], clip_id=it["clip_id"],
                frame_width=img.shape[1], frame_height=img.shape[0],
                corners=res, visible=res is not None, label_source="manual",
            )) + "\n")
            f.flush()
            print(f"[{n + 1}/{len(queue)}] {it['path'].name}: {'ok' if res else 'skipped'}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
