"""CP16 step 16a: training windows for the MMAudio LoRA fine-tune (env guitar2midi).

    python scripts/34a_make_lora_windows.py

GAPS **train** split only (test never used). Every 8 s window that fits inside the downloaded video
(``mmaudio_lora.make_windows``); notes per window counted in the GAPS MIDI (video-time lag of 0-49 ms ignored here,
it only matters for counting); windows with fewer than --min-notes notes are marked ``keep=0`` (intros, talking,
applause, silence). Validation part = ~10 % of the pieces, whole performers (``mmaudio_lora.performer_split``).
Also reports which test-split performers appear in train (GAPS splits are by piece, not by performer).

Writes data/processed/mmaudio_lora/{windows.csv, split.csv, summary.txt}.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pretty_midi

from guitar2midi.baselines import mmaudio_lora as L

GAPS = Path("data/raw/gaps")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/processed/mmaudio_lora"))
    p.add_argument("--min-notes", type=int, default=3)
    p.add_argument("--val-share", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train = list(csv.DictReader(open(GAPS / "summary_train.csv")))
    test = list(csv.DictReader(open(GAPS / "summary_test.csv")))
    pieces = [r for r in train if r["video_s"] and (GAPS / "videos" / f"{r['id']}.mp4").exists()]
    val = L.performer_split(pieces, args.val_share, args.seed)

    rows = []
    for r in pieces:
        midi = pretty_midi.PrettyMIDI(str(GAPS / "midi" / f"{r['id']}.mid"))
        on = np.array(sorted(n.start for i in midi.instruments for n in i.notes))
        for s in L.make_windows(float(r["video_s"])):
            n = int(((on >= s) & (on < s + L.WINDOW_S)).sum())
            rows.append({"window_id": f"{r['id']}_{int(round(s * 1000)):07d}", "piece": r["id"], "start_s": s,
                         "split": "val" if r["id"] in val else "train", "performer": r["performer"],
                         "n_notes": n, "keep": int(n >= args.min_notes)})
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["piece", "performer", "split", "video_s"])
        for r in pieces:
            w.writerow([r["id"], r["performer"], "val" if r["id"] in val else "train", r["video_s"]])

    lines = [f"GAPS train pieces: {len(train)}, with video: {len(pieces)} "
             f"({sum(float(r['video_s']) for r in pieces) / 3600:.2f} h)"]
    for sp in ("train", "val"):
        ps = [r for r in pieces if ("val" if r["id"] in val else "train") == sp]
        ws = [r for r in rows if r["split"] == sp]
        kept = [r for r in ws if r["keep"]]
        perf = {r["performer"] or r["id"] for r in ps}
        lines.append(f"  {sp:5s}: {len(ps):3d} pieces, {len(perf):3d} performers, {len(ws):5d} windows, "
                     f"{len(kept):5d} kept ({len(kept) * L.WINDOW_S / 3600:.2f} h), "
                     f"notes per kept window median {np.median([r['n_notes'] for r in kept]):.0f}")
    lines.append(f"  dropped (< {args.min_notes} notes): {sum(not r['keep'] for r in rows)} windows")
    val_perf = {r["performer"] for r in pieces if r["id"] in val}
    train_perf = {r["performer"] for r in pieces if r["id"] not in val}
    lines.append(f"  performers in both train and val: {sorted((val_perf & train_perf) - {'', 'Unknown'}) or 'none'}")
    seen = sorted({r["performer"] for r in test} & {r["performer"] for r in pieces} - {"", "Unknown"})
    lines.append(f"Test performers also in GAPS train (GAPS splits are by piece): {len(seen)} of "
                 f"{len({r['performer'] for r in test})}; test pieces by them: "
                 f"{sum(r['performer'] in seen for r in test)} of {len(test)}")
    lines.append("  " + ", ".join(seen))
    top = Counter(r["performer"] for r in rows if r["keep"]).most_common(3)
    lines.append(f"Largest performers (kept windows): {top}")
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
