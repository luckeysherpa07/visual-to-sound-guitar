"""CP16 step 16d: LoRA A / B / B-blank vs MMAudio zero-shot on the evaluation clips (env guitar2midi).

    python scripts/34g_compare_lora_eval.py

Reads the per-clip scores of the 30e runs outputs/runs/<date>_cp16_lora_{gaps,guifir,magcil}/per_clip.csv
(masked clips, same as every baseline row). For every dataset and prompt setting:
  - mean note F1@50, timing F1, chroma similarity, frame F1 per system,
  - paired Wilcoxon tests over clips: A vs zero-shot, A vs B (+video), A vs B blank, B (+video) vs zero-shot.
GAPS also split into clips by performers that are / are not in GAPS train (seen: 12 of 27 test performers).

Writes outputs/runs/<today>_cp16_lora_compare/summary.md.
"""

import argparse
import csv
import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RUNS = Path("outputs/runs")
SETS = ("gaps", "guifir", "magcil")
METRICS = ("onset_f50", "timing_f50", "chroma_sim", "frame_f")
SYS = {"zero-shot": "mmaudio", "A": "mmaudio_loraA", "B (+video)": "mmaudio_loraB", "B blank": "mmaudio_loraBblank"}
PAIRS = [("A", "zero-shot"), ("A", "B (+video)"), ("A", "B blank"), ("B (+video)", "zero-shot")]


def latest(name: str) -> Path:
    return sorted(RUNS.glob(f"*_{name}"))[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=RUNS / f"{datetime.date.today()}_cp16_lora_compare")
    args = p.parse_args()

    train_perf = {r["performer"] for r in csv.DictReader(open("data/raw/gaps/summary_train.csv"))} - {"", "Unknown"}
    perf = {r["clip_id"]: r["performer"] for r in csv.DictReader(open("data/interim/baseline_clips/clips.csv"))}
    md = ["# CP16 16d: MMAudio LoRA vs zero-shot (masked evaluation clips)", "",
          "Scores in % (chroma in [0, 1]); tests = paired Wilcoxon over clips, 'better' = share of clips where the "
          "first system scores higher.", ""]
    for ds in SETS:
        run = latest(f"cp16_lora_{ds}")
        rows = list(csv.DictReader(open(run / "per_clip.csv")))
        score = {}
        for r in rows:
            score.setdefault(r["system"], {})[r["clip_id"]] = r
        subsets = {"all": None}
        if ds == "gaps":
            subsets["seen performers"] = {c for c, pf in perf.items() if pf in train_perf}
            subsets["unseen performers"] = {c for c, pf in perf.items() if pf not in train_perf}
        for prompt in ("noprompt", "prompt"):
            for sub, keep in subsets.items():
                clips = sorted(c for c in score["real_audio"] if keep is None or c in keep)
                md += [f"## {ds}, {prompt}, {sub} ({len(clips)} clips; run {run.name})", "",
                       "| system | " + " | ".join(METRICS) + " | other-video note F1 |", "|---" * (len(METRICS) + 2) + "|"]
                for label, base in SYS.items():
                    s = score.get(f"{base}_{prompt}")
                    if s is None:
                        continue
                    sh = score.get(f"{base}_{prompt}@shuffled", {})
                    vals = [np.mean([float(s[c][m]) for c in clips]) for m in METRICS]
                    shv = np.mean([float(sh[c]["onset_f50"]) for c in clips]) * 100 if sh else float("nan")
                    md.append(f"| {label} | " + " | ".join(f"{v:.3f}" if m == "chroma_sim" else f"{100 * v:.1f}"
                                                           for v, m in zip(vals, METRICS)) + f" | {shv:.1f} |")
                ch = score.get("real_audio@shuffled")
                if ch:
                    md.append(f"| chance (real audio, other clip) | {100 * np.mean([float(ch[c]['onset_f50']) for c in clips]):.1f}"
                              + " |" * (len(METRICS) + 1))
                md += ["", "| test | metric | first | second | better | p |", "|---|---|---|---|---|---|"]
                for a, b in PAIRS:
                    sa, sb = score.get(f"{SYS[a]}_{prompt}"), score.get(f"{SYS[b]}_{prompt}")
                    if not sa or not sb:
                        continue
                    for m in ("onset_f50", "timing_f50", "chroma_sim"):
                        x = np.array([float(sa[c][m]) for c in clips])
                        y = np.array([float(sb[c][m]) for c in clips])
                        pv = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
                        md.append(f"| {a} vs {b} | {m} | {x.mean():.3f} | {y.mean():.3f} | {np.mean(x > y):.2f} | {pv:.1e} |")
                md.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
