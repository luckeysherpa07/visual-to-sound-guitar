"""CP16: training / validation curves of LoRA runs (env guitar2midi).

    python scripts/34e_plot_lora_curves.py outputs/runs/<run> [outputs/runs/<run2> ...] --out curves.png

One panel per run, from its log.csv (34d): mean training loss since the last validation, validation loss with the
window's own video, another piece's video, and a blank video (fixed noise, no condition dropout).
Default output: <first run>/curves.png.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    fig, axes = plt.subplots(1, len(args.runs), figsize=(6 * len(args.runs), 4), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], args.runs):
        rows = list(csv.DictReader(open(run / "log.csv")))
        cfg = json.loads((run / "config.json").read_text())
        step = [int(r["step"]) for r in rows]
        for key, label, style in [("train_loss", "train (mean since last point)", "k-"),
                                  ("val_own", "val, own video", "C0-o"), ("val_other", "val, another piece's video", "C3--"),
                                  ("val_blank", "val, blank video", "C2:")]:
            ax.plot(step, [float(r[key]) for r in rows], style, label=label, ms=3)
        what = f"{cfg['n_train']} windows" + (" (train = val)" if cfg.get("sanity") else f", val {cfg['n_val']}")
        ax.set_title(f"{run.name}\nlr {cfg['lr']:g}, rank {cfg['rank']}, batch {cfg['batch']}, {what}"
                     + (", video blanked" if cfg.get("no_video") else ""), fontsize=9)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("flow-matching loss")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    out = args.out or args.runs[0] / "curves.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
