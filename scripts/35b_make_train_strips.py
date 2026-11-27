"""CP17 step 17b: track the clicked GAPS train/val shots, write both strip versions + labels in V2N's format.

    python scripts/35b_make_train_strips.py                    # all clicked pieces (resumes finished ones)
    python scripts/35b_make_train_strips.py --pieces 001_mvswc 003_HSswc

Inputs: data/interim/v2n_train/shots.json (35a), corner clicks data/labels/fretboard_corners/gaps_train.jsonl
(01b, record clip_id = <piece>__s<k>), split data/processed/mmaudio_lora/split.csv (CP16; train -> V2N split
"train", val -> "valid"), GAPS wav + MIDI + video. Method and formats: guitar2midi.baselines.v2n_data.

Writes under --out (default data/processed/v2n_guitar/):
    {fret,full}/pieces/<piece>.h5                   records of one piece (JPEG strips; fret = nut -> fret 20 (CP5),
                                                    full = nut -> bridge (CP12)), same frames in both
    {fret,full}/processed_v2/{train,valid}/guitar_v2.hdf5   per split: external links to the piece files (V2N's
                                                    data_dir = <out>/<version>, dataset.data_dir = processed_v2)
    TSV/<record>.tsv                                labels (shared by both versions)
    meta/<piece>.json                               sync, shots, records of one piece (marks the piece as done)
    check/<record>.jpg                              frame + tracked quad + both strips every 30 s of the record
    records.csv, pieces.csv, drift_sheet.jpg (the longest records), summary.txt
"""

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from guitar2midi.baselines import v2n_data as D

SHOTS = Path("data/interim/v2n_train/shots.json")
ANN = Path("data/labels/fretboard_corners/gaps_train.jsonl")
SPLIT = Path("data/processed/mmaudio_lora/split.csv")
GAPS = Path("data/raw/gaps")
OUT = Path("data/processed/v2n_guitar")
V2N_SPLIT = {"train": "train", "val": "valid"}


def load_annotations(path: Path) -> dict:
    anns = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            anns[r["clip_id"]] = r  # last record wins (re-annotation)
    return anns


def process(piece: str, info: dict, anns: dict, out: Path) -> dict:
    video = GAPS / "videos" / f"{piece}.mp4"
    meta = {"piece": piece, "shots": [], "records": []}
    w, h = D.video_size(video)
    shots = []
    for s in info["chosen"]:
        name = f"{piece}__s{s['shot']}"
        r = anns.get(name)
        if r is None:
            meta["shots"].append({"shot": name, "status": "not clicked"})
            continue
        if not r["visible"]:
            meta["shots"].append({"shot": name, "status": "skipped (no usable fretboard)"})
            continue
        ref = int(re.search(r"f(\d+)\.png$", r["image"]).group(1))
        corners = np.array(r["corners"], float)
        corners *= [w / r["frame_width"], h / r["frame_height"]]  # 1.0 unless the click image was resized
        shots.append(D.Shot(name, s["start"], s["end"], ref, corners))
    if not shots:
        meta["status"] = "no clicked shots"
        return meta

    sync = D.piece_sync(GAPS / "audio" / f"{piece}.wav", video, info["n_frames"] / D.FPS)
    meta["sync"] = sync
    if not sync["ok"]:
        meta["status"] = "sync failed"
        return meta

    n, _ = D.track_shots(video, shots)
    for s in shots:
        cov = float(s.tracked.mean()) if s.tracked is not None else 0.0
        kept = sum(b - a for a, b in s.runs)
        # stretches of the shot without a record, >= 10 s: candidates for a second click round
        bounds = [s.start, *[x for r in s.runs for x in r], s.end]
        uncovered = [(a, b) for a, b in zip(bounds[::2], bounds[1::2]) if b - a >= 10 * D.FPS]
        meta["shots"].append({"shot": s.name, "status": "ok", "frames": s.end - s.start, "tracked": round(cov, 3),
                              "kept_frames": kept, "n_records": len(s.runs), "uncovered": uncovered})
    notes = D.load_notes(GAPS / "midi" / f"{piece}.mid")
    rows, checks = D.write_records(video, shots, piece, out, out / "TSV", notes, sync["lag"])
    meta["records"] = rows
    meta["status"] = "ok" if rows else "no records"
    (out / "check").mkdir(parents=True, exist_ok=True)
    by_rec = {}
    for rec, img in checks:
        by_rec.setdefault(rec, []).append(img)
    for rec, imgs in by_rec.items():
        imgs = imgs[:10]
        hmax = max(i.shape[0] for i in imgs)
        imgs = [np.vstack([i, np.zeros((hmax - i.shape[0], i.shape[1], 3), np.uint8)]) for i in imgs]
        cv2.imwrite(str(out / "check" / f"{rec}.jpg"), np.hstack(imgs), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return meta


def run(args_tuple):
    piece, info, anns, out = args_tuple
    meta_file = out / "meta" / f"{piece}.json"
    try:
        meta = process(piece, info, anns, out)
    except Exception as e:  # keep going; the piece is retried on the next run
        return {"piece": piece, "status": f"error: {e!r}"}
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    D.save_json(meta, meta_file)
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shots", type=Path, default=SHOTS)
    p.add_argument("--annotations", type=Path, default=ANN)
    p.add_argument("--split-csv", type=Path, default=SPLIT)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--pieces", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=3, help="each worker holds one piece's shots (<= ~5 GB)")
    p.add_argument("--redo", action="store_true", help="reprocess pieces that already have meta/<piece>.json")
    args = p.parse_args()
    out = args.out

    shots = json.loads(args.shots.read_text())
    anns = load_annotations(args.annotations)
    with open(args.split_csv) as f:
        split_of = {r["piece"]: r["split"] for r in csv.DictReader(f) if r["split"] in V2N_SPLIT}
    clicked = {a.split("__")[0] for a in anns}
    pieces = [pc for pc in (args.pieces or sorted(shots)) if pc in clicked and pc in split_of]
    todo = [pc for pc in pieces if args.redo or not (out / "meta" / f"{pc}.json").exists()]
    print(f"{len(pieces)} pieces with clicks, {len(todo)} to process", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        for i, meta in enumerate(ex.map(run, [(pc, shots[pc], {k: v for k, v in anns.items()
                                                                if k.startswith(pc + "__")}, out) for pc in todo])):
            nrec = len(meta.get("records", []))
            sec = sum(r["seconds"] for r in meta.get("records", []))
            print(f"[{i + 1}/{len(todo)}] {meta['piece']}: {meta['status']}, {nrec} records, {sec / 60:.1f} min"
                  + (f", lag {meta['sync']['lag'] * 1000:.0f} ms" if "sync" in meta else ""), flush=True)

    # ---- collect all finished pieces (also from earlier runs)
    metas = [json.loads((out / "meta" / f"{pc}.json").read_text()) for pc in pieces
             if (out / "meta" / f"{pc}.json").exists()]
    rec_rows, piece_rows = [], []
    for m in metas:
        recs = m.get("records", [])
        for r in recs:
            r["split"] = V2N_SPLIT[split_of[m["piece"]]]
        rec_rows += recs
        shots_ok = [s for s in m["shots"] if s.get("status") == "ok"]
        piece_rows.append({"piece": m["piece"], "split": V2N_SPLIT[split_of[m["piece"]]], "status": m["status"],
                           "lag_ms": round(m["sync"]["lag"] * 1000, 1) if "sync" in m else "",
                           "sync_min_corr": round(m["sync"]["min_corr"], 3) if "sync" in m else "",
                           "shots_clicked": len(shots_ok),
                           "shots_skipped": sum(s["status"].startswith("skipped") for s in m["shots"]),
                           "shot_minutes": round(sum(s["frames"] for s in shots_ok) / D.FPS / 60, 2),
                           "kept_minutes": round(sum(r["seconds"] for r in recs) / 60, 2),
                           "n_records": len(recs)})
    for name, rows in (("records.csv", rec_rows), ("pieces.csv", piece_rows)):
        if rows:
            with open(out / name, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)

    lines = []
    for v in D.VERSIONS:
        for split in ("train", "valid"):
            files = {r["record"]: out / v / "pieces" / f"{r['piece']}.h5" for r in rec_rows if r["split"] == split}
            if files:
                D.build_master(files, out / v / "processed_v2" / split / D.H5_NAME)
    for split in ("train", "valid"):
        rs = [r for r in rec_rows if r["split"] == split]
        ps = {r["piece"] for r in rs}
        lines.append(f"{split}: {len(ps)} pieces, {len(rs)} records, {sum(r['seconds'] for r in rs) / 3600:.2f} h, "
                     f"{sum(r['onsets_inside'] for r in rs)} note onsets, lost (bridged) frames "
                     f"{sum(r['lost_frames'] for r in rs)}")
    bad = [r for r in piece_rows if r["status"] != "ok"]
    lines.append(f"pieces not ok: {len(bad)} " + ", ".join(f"{r['piece']} ({r['status']})" for r in bad))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # ---- drift sheet: the 8 longest records (tracking over the longest time from the click)
    longest = sorted(rec_rows, key=lambda r: -r["n_frames"])[:8]
    rows_img = []
    for r in longest:
        f = out / "check" / f"{r['record']}.jpg"
        if f.exists():
            im = cv2.imread(str(f))
            im = im[:, : min(im.shape[1], 2000)]
            rows_img.append(cv2.resize(im, (2000, int(im.shape[0] * 2000 / im.shape[1])))
                            if im.shape[1] != 2000 else im)
    if rows_img:
        cv2.imwrite(str(out / "drift_sheet.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
