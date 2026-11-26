"""CP16 steps 16b/16c: LoRA fine-tune of MMAudio on the GAPS train windows (env mmaudio).

    # 16b sanity run: 8 windows (1 from each of 8 train pieces), train = val, must almost memorise them
    PYTHONNOUSERSITE=1 python scripts/34d_train_lora.py --run-name cp16_lora_sanity --sanity 8 --steps 400 \\
        --val-every 25 --check-merge
    # 16c run A (video + text) / run B (text only, video blanked)
    PYTHONNOUSERSITE=1 python scripts/34d_train_lora.py --run-name cp16_lora_A
    PYTHONNOUSERSITE=1 python scripts/34d_train_lora.py --run-name cp16_lora_B --no-video

Input:  data/processed/mmaudio_lora/{split.csv, features/, text_features.npy} (34a, 34b)
Writes: outputs/runs/<date>_<run-name>/  config.json, log.csv (step, lr, train loss, val loss with own / another
        piece's / blank video), adapters_best.pt (lowest val loss with own video; run B: blank), adapters_last.pt,
        summary.txt. Plot with 34e_plot_lora_curves.py (env guitar2midi).

Training = MMAudio's own train_fn (log-normal noise level t, flow-matching loss, video and text each "empty" 10 % of
the time) and optimiser settings (AdamW betas 0.9/0.95, eps 1e-6, weight decay 1e-6, grad-norm clip 1.0,
bf16 autocast, linear warm-up), but only the LoRA adapters are trained (base model frozen).
"""

import argparse
import csv
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch

from guitar2midi.baselines import mmaudio_lora as L

D = Path("data/processed/mmaudio_lora")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--no-video", action="store_true", help="control run B: video always blanked")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, help="LoRA scale alpha/rank; default alpha = rank")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--sanity", type=int, help="train and validate on this many windows (1 per train piece)")
    p.add_argument("--check-merge", action="store_true", help="after training: merged model gives the same loss")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run = Path("outputs/runs") / f"{datetime.date.today()}_{args.run_name}"
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    split = list(csv.DictReader(open(D / "split.csv")))
    train_p = sorted(r["piece"] for r in split if r["split"] == "train")
    val_p = sorted(r["piece"] for r in split if r["split"] == "val")
    if args.sanity:
        pick = list(np.random.default_rng(args.seed).choice(train_p, args.sanity, replace=False))
        train = val = L.load_features(D / "features", sorted(pick), per_piece=1)
    else:
        train, val = L.load_features(D / "features", train_p), L.load_features(D / "features", val_p)
    print(f"train {len(train['ids'])} windows, val {len(val['ids'])} windows", flush=True)

    tr = L.LoRATrainer(np.load(D / "text_features.npy"), rank=args.rank, alpha=args.alpha, video=not args.no_video,
                       seed=args.seed)
    print(f"MMAudio {tr.n_total / 1e6:.1f} M parameters (frozen); LoRA on {len(tr.wrapped)} layers: "
          f"{tr.n_lora / 1e6:.2f} M trainable ({100 * tr.n_lora / tr.n_total:.2f} %)", flush=True)
    opt = torch.optim.AdamW(tr.params, lr=args.lr, betas=(0.9, 0.95), eps=1e-6, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    cfg = vars(args) | {"n_train": len(train["ids"]), "n_val": len(val["ids"]), "n_lora": tr.n_lora,
                        "lora_layers": len(tr.wrapped), "targets": L.LORA_TARGETS, "null_cond_p": L.NULL_COND_P,
                        "train_ids": train["ids"] if args.sanity else None}
    (run / "config.json").write_text(json.dumps(cfg, indent=1))

    main_mode = "blank" if args.no_video else "own"
    fields = ["step", "lr", "train_loss", "val_own", "val_other", "val_blank", "min_per_step"]
    log = open(run / "log.csv", "w", newline="")
    w = csv.DictWriter(log, fieldnames=fields)
    w.writeheader()

    def validate(step: int, train_loss: float, dt: float) -> float:
        row = {"step": step, "lr": sched.get_last_lr()[0], "train_loss": train_loss, "min_per_step": dt}
        for mode in ("own", "other", "blank"):
            row[f"val_{mode}"] = float(tr.eval_loss(val, video_mode=mode).mean())
        w.writerow(row)
        log.flush()
        print(f"step {step:5d}  train {train_loss:.4f}  val own {row['val_own']:.4f}  other {row['val_other']:.4f}  "
              f"blank {row['val_blank']:.4f}  ({dt * 1000:.0f} ms/step, "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
        return row[f"val_{main_mode}"]

    best = validate(0, float("nan"), 0.0)
    best_step = 0
    torch.save(tr.adapters(), run / "adapters_best.pt")
    n, order, pos, losses, t0 = len(train["ids"]), torch.randperm(len(train["ids"])), 0, [], time.time()
    for step in range(1, args.steps + 1):
        if pos + args.batch > n:
            order, pos = torch.randperm(n), 0
        idx = order[pos:pos + args.batch] if n >= args.batch else torch.randint(0, n, (args.batch,))
        pos += args.batch
        loss = tr.loss(train, idx).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tr.params, 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % args.val_every == 0 or step == args.steps:
            v = validate(step, float(np.mean(losses)), (time.time() - t0) / len(losses))
            losses, t0 = [], time.time()
            if v < best:
                best, best_step = v, step
                torch.save(tr.adapters(), run / "adapters_best.pt")
    torch.save(tr.adapters(), run / "adapters_last.pt")
    log.close()

    lines = [f"best val loss ({main_mode} video) {best:.4f} at step {best_step}"]
    if args.check_merge:
        before = tr.eval_loss(val, video_mode=main_mode)
        k = tr.lora.merge_lora(tr.net)
        after = tr.eval_loss(val, video_mode=main_mode)
        lines.append(f"merge check: {k} layers merged, val loss {before.mean():.5f} -> {after.mean():.5f} "
                     f"(max per-window diff {np.abs(before - after).max():.5f})")
    (run / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
