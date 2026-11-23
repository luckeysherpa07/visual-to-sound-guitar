"""Evaluate transcriptions of the baseline clips against GAPS ground truth (CP3, CP7).

    python scripts/30e_evaluate_baselines.py --run-name cp3_ceiling
    python scripts/30e_evaluate_baselines.py --run-name cp7_v2a --systems real_audio mmaudio_noprompt silence \
        --shuffled real_audio mmaudio_noprompt

Systems = folders in data/interim/baseline_clips/transcriptions/<system>/<clip>.mid (default: all of
them), plus the built-in "silence" (no notes). Only clips with keep=1 in clips.csv are used.

--shuffled S: adds system "S@shuffled", where each clip is scored against S's output for a clip of a
*different piece* (fixed pairing: the clip 3 positions later in clips.csv, i.e. the next piece). For a V2A
system this shows whether the video matters at all; for real_audio it is the score of "real guitar music,
wrong piece" (chance level for the metrics).

--mask-dir DIR (e.g. data/interim/baseline_clips/strips): use only clips that have DIR/<clip>_valid.npy and
score only notes/frames where it is True (camera cuts, dissolves). Applied to every system, so all are
compared on exactly the same clips and time.

Writes outputs/runs/<YYYY-MM-DD>_<run-name>/{per_clip.csv, summary.csv, summary.md, args.json}.
"""

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from guitar2midi.evaluation import note_metrics as M

CLIPS = Path("data/interim/baseline_clips")
MAIN = ["onset_f50", "onset_p50", "onset_r50", "onset_f100", "onset_offset_f50", "frame_f", "timing_f50",
        "chroma_sim", "onset_corr", "n_est"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--systems", nargs="+", help="default: every transcription folder + silence")
    p.add_argument("--shuffled", nargs="+", default=[], help="systems to also score with mismatched clips")
    p.add_argument("--clips", type=Path, default=CLIPS)
    p.add_argument("--mask-dir", type=Path, help="restrict to clips/frames with <clip>_valid.npy here")
    p.add_argument("--length", type=float, default=8.0)
    args = p.parse_args()

    trans = args.clips / "transcriptions"
    systems = args.systems or sorted(d.name for d in trans.iterdir() if d.is_dir()) + ["silence"]
    with open(args.clips / "clips.csv") as f:
        clips = [r for r in csv.DictReader(f) if r["keep"] == "1"]
    masks = {}
    if args.mask_dir:
        masks = {c["clip_id"]: np.load(args.mask_dir / f"{c['clip_id']}_valid.npy") for c in clips
                 if (args.mask_dir / f"{c['clip_id']}_valid.npy").exists()}
        clips = [c for c in clips if c["clip_id"] in masks]
        print(f"masked evaluation: {len(clips)} clips, "
              f"{np.mean([m.mean() for m in masks.values()]):.1%} of their frames valid")

    def partner(i: int) -> dict:
        """First clip >= 3 positions later (cyclic) that belongs to another piece."""
        for k in range(3, len(clips) + 3):
            other = clips[(i + k) % len(clips)]
            if other["piece_id"] != clips[i]["piece_id"]:
                return other
        raise ValueError("need clips from at least two pieces")

    systems = systems + [f"{s}@shuffled" for s in args.shuffled]
    rows = []
    for system in systems:
        missing = 0
        for i, c in enumerate(clips):
            ref = M.load_notes(args.clips / "midi" / f"{c['clip_id']}.mid")
            src_system, src_clip = system, c["clip_id"]
            if system.endswith("@shuffled"):
                src_system, src_clip = system[: -len("@shuffled")], partner(i)["clip_id"]
            if system == "silence":
                est = np.zeros((0, 3))
            else:
                f = trans / src_system / f"{src_clip}.mid"
                if not f.exists():
                    missing += 1
                    continue
                est = M.load_notes(f)
            rows.append({"system": system, "clip_id": c["clip_id"], "piece_id": c["piece_id"],
                         **M.evaluate(ref, est, args.length, valid=masks.get(c["clip_id"]))})
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

    pct = {"onset_f50", "onset_p50", "onset_r50", "onset_f100", "onset_offset_f50", "frame_f", "timing_f50"}
    header = "| system | clips | " + " | ".join(MAIN) + " |"
    lines = [header, "|" + "---|" * (len(MAIN) + 2)]
    for s in summary:
        cells = [f"{100 * s[k]:.1f}" if k in pct else (f"{s[k]:.1f}" if k == "n_est" else f"{s[k]:.3f}")
                 for k in MAIN]
        lines.append(f"| {s['system']} | {s['clips']} | " + " | ".join(cells) + " |")
    md = ("Mean over clips. F1/P/R/frame in %, onset tolerance 50/100 ms (mir_eval, onsets only unless "
          "'offset'), frame F1 at 25 fps, timing_f50 = pluck-time F1 ignoring pitch; chroma_sim and onset_corr in [0, 1]; n_est = notes per clip "
          f"(GT mean {np.mean([r['n_ref'] for r in rows]):.1f}).\n\n" + "\n".join(lines) + "\n")
    # matched vs shuffled: does the system's output depend on *this* clip? (paired over clips)
    tests = []
    for base in args.shuffled:
        a = {r["clip_id"]: r for r in rows if r["system"] == base}
        b = {r["clip_id"]: r for r in rows if r["system"] == f"{base}@shuffled"}
        ids = sorted(a.keys() & b.keys())
        for k in ("onset_f50", "timing_f50", "chroma_sim", "onset_corr"):
            x = np.array([a[i][k] for i in ids], float)
            y = np.array([b[i][k] for i in ids], float)
            pval = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
            tests.append(f"| {base} | {k} | {x.mean():.3f} | {y.mean():.3f} | {np.mean(x > y):.2f} | {pval:.1e} |")
    if tests:
        md += ("\nMatched vs shuffled (paired Wilcoxon signed-rank over clips; 'better' = share of clips where "
               "matched > shuffled):\n\n| system | metric | matched | shuffled | better | p |\n|---|---|---|---|---|---|\n"
               + "\n".join(tests) + "\n")
    (out / "summary.md").write_text(md)
    (out / "args.json").write_text(json.dumps({**vars(args), "clips": str(args.clips), "systems": systems,
                                               "mask_dir": str(args.mask_dir) if args.mask_dir else None}, indent=2))
    print(md)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
