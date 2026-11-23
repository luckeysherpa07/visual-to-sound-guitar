"""Evaluate transcriptions of the baseline clips against GAPS ground truth (CP3, CP7).

    python scripts/30e_evaluate_baselines.py --run-name cp3_ceiling
    python scripts/30e_evaluate_baselines.py --run-name cp7_v2a --systems real_audio mmaudio_noprompt silence

Systems = folders in data/interim/baseline_clips/transcriptions/<system>/<clip>.mid (default: all of
them), plus the built-in "silence" (no notes). Only clips with keep=1 in clips.csv are used.

Writes outputs/runs/<YYYY-MM-DD>_<run-name>/{per_clip.csv, summary.csv, summary.md, args.json}.
"""

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from guitar2midi.evaluation import note_metrics as M

CLIPS = Path("data/interim/baseline_clips")
MAIN = ["onset_f50", "onset_p50", "onset_r50", "onset_f100", "onset_offset_f50", "frame_f", "chroma_sim",
        "onset_corr", "n_est"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--systems", nargs="+", help="default: every transcription folder + silence")
    p.add_argument("--clips", type=Path, default=CLIPS)
    p.add_argument("--length", type=float, default=8.0)
    args = p.parse_args()

    trans = args.clips / "transcriptions"
    systems = args.systems or sorted(d.name for d in trans.iterdir() if d.is_dir()) + ["silence"]
    with open(args.clips / "clips.csv") as f:
        clips = [r for r in csv.DictReader(f) if r["keep"] == "1"]

    rows = []
    for system in systems:
        missing = 0
        for c in clips:
            ref = M.load_notes(args.clips / "midi" / f"{c['clip_id']}.mid")
            if system == "silence":
                est = np.zeros((0, 3))
            else:
                f = trans / system / f"{c['clip_id']}.mid"
                if not f.exists():
                    missing += 1
                    continue
                est = M.load_notes(f)
            rows.append({"system": system, "clip_id": c["clip_id"], "piece_id": c["piece_id"],
                         **M.evaluate(ref, est, args.length)})
        if missing:
            print(f"WARNING: {system}: {missing} clips without transcription (skipped)")

    out = Path("outputs/runs") / f"{date.today().isoformat()}_{args.run_name}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for system in systems:
        rs = [r for r in rows if r["system"] == system]
        if not rs:
            continue
        s = {"system": system, "clips": len(rs)}
        for k in MAIN:
            v = np.array([r[k] for r in rs], float)
            s[k] = round(float(v.mean()), 4)
            s[f"{k}_std"] = round(float(v.std()), 4)
        summary.append(s)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    pct = {"onset_f50", "onset_p50", "onset_r50", "onset_f100", "onset_offset_f50", "frame_f"}
    header = "| system | clips | " + " | ".join(MAIN) + " |"
    lines = [header, "|" + "---|" * (len(MAIN) + 2)]
    for s in summary:
        cells = [f"{100 * s[k]:.1f}" if k in pct else (f"{s[k]:.1f}" if k == "n_est" else f"{s[k]:.3f}")
                 for k in MAIN]
        lines.append(f"| {s['system']} | {s['clips']} | " + " | ".join(cells) + " |")
    md = ("Mean over clips. F1/P/R/frame in %, onset tolerance 50/100 ms (mir_eval, onsets only unless "
          "'offset'), frame F1 at 25 fps; chroma_sim and onset_corr in [0, 1]; n_est = notes per clip "
          f"(GT mean {np.mean([r['n_ref'] for r in rows]):.1f}).\n\n" + "\n".join(lines) + "\n")
    (out / "summary.md").write_text(md)
    (out / "args.json").write_text(json.dumps({**vars(args), "clips": str(args.clips), "systems": systems}, indent=2))
    print(md)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
