"""Train the fretboard pose estimator (MMPose). Run from the repo root.

    python scripts/01d_train_fpe.py --run-name fpe_res50
    # sanity check: overfit 50 training images for 5 epochs
    python scripts/01d_train_fpe.py --run-name fpe_sanity \
        --cfg-options train_dataloader.dataset.indices=50 train_cfg.max_epochs=5 train_cfg.val_interval=1
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/preprocessing/fpe_res50_regression.py")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = str(Path("outputs/runs") / f"{date.today().isoformat()}_{args.run_name}")
    cfg.resume = args.resume
    Runner.from_cfg(cfg).train()  # mmengine dumps the merged config into work_dir


if __name__ == "__main__":
    main()
