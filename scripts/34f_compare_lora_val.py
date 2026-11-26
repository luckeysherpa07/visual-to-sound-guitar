"""CP16 step 16c: per-window validation loss of MMAudio zero-shot vs LoRA run A vs run B (env mmaudio).

    PYTHONNOUSERSITE=1 python scripts/34f_compare_lora_val.py

Same 586 validation windows (unseen performers), same fixed noise draws as the curves (34d eval_loss), for
    zero-shot   MMAudio without adapters
    A           adapters_best.pt of run A (video + text)
    B           adapters_best.pt of run B (text only, video blanked)
each with own video / another piece's video / blank video. Paired Wilcoxon tests per window answer:
    does A beat B with the video?  (= did the video part of the training add anything beyond "guitar sound")
    is A's own-vs-blank / own-vs-other gap larger than zero-shot's?  (= does A use the video more than before)

Writes outputs/runs/2026-11-26_cp16_lora_val_compare/{per_window.csv, summary.txt}.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from guitar2midi.baselines import mmaudio_lora as L

D = Path("data/processed/mmaudio_lora")
RUNS = {"A": Path("outputs/runs/2026-11-26_cp16_lora_A"), "B": Path("outputs/runs/2026-11-26_cp16_lora_B")}
OUT = Path("outputs/runs/2026-11-26_cp16_lora_val_compare")
MODES = ("own", "other", "blank")


def main() -> None:
    val_p = sorted(r["piece"] for r in csv.DictReader(open(D / "split.csv")) if r["split"] == "val")
    val = L.load_features(D / "features", val_p)
    tr = L.LoRATrainer(np.load(D / "text_features.npy"), rank=16)  # adapters start at B = 0 = zero-shot
    res = {}
    for mode in MODES:
        res[f"zeroshot_{mode}"] = tr.eval_loss(val, video_mode=mode)
    for name, run in RUNS.items():
        tr.lora.load_lora(tr.net, torch.load(run / "adapters_best.pt", map_location="cuda", weights_only=True))
        for mode in MODES:
            res[f"{name}_{mode}"] = tr.eval_loss(val, video_mode=mode)
        print(name, {m: round(float(res[f"{name}_{m}"].mean()), 4) for m in MODES}, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "per_window.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id"] + list(res))
        for i, wid in enumerate(val["ids"]):
            w.writerow([wid] + [f"{res[k][i]:.5f}" for k in res])

    def test(label: str, a: np.ndarray, b: np.ndarray) -> str:
        d = a - b
        return (f"{label:52s} mean diff {d.mean():+.4f}  ({(d < 0).mean() * 100:4.1f} % of windows lower)  "
                f"p = {wilcoxon(a, b).pvalue:.1e}")

    z, A, B = (lambda m, n=n: res[f"{n}_{m}"] for n in ("zeroshot", "A", "B"))
    lines = [f"{len(val['ids'])} validation windows, mean flow-matching loss (fixed noise draws):",
             f"{'':10s}" + "".join(f"{m + ' video':>14s}" for m in MODES)]
    for n in ("zeroshot", "A", "B"):
        lines.append(f"{n:10s}" + "".join(f"{res[f'{n}_{m}'].mean():14.4f}" for m in MODES))
    lines += ["", "paired tests (a - b per window; negative = a better):",
              test("A own video  vs  zero-shot own video", A("own"), z("own")),
              test("B (blank)    vs  zero-shot blank", B("blank"), z("blank")),
              test("A own video  vs  B (blank)", A("own"), B("blank")),
              test("A: own vs blank video", A("own"), A("blank")),
              test("zero-shot: own vs blank video", z("own"), z("blank")),
              test("A own-blank gap  vs  zero-shot own-blank gap", A("own") - A("blank"), z("own") - z("blank")),
              test("A: own vs another piece's video", A("own"), A("other")),
              test("zero-shot: own vs another piece's video", z("own"), z("other")),
              test("A own-other gap  vs  zero-shot own-other gap", A("own") - A("other"), z("own") - z("other"))]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
