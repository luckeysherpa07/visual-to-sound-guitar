"""CP17 step 17c/17d: train V2N on the GAPS guitar strips (env v2n). V2N's own trainer, model and data code.

    python scripts/35c_train_v2n.py --run A            # piano init, fretboard strips (main run)
    python scripts/35c_train_v2n.py --run D            # piano init, full-string strips (picking hand visible)
    python scripts/35c_train_v2n.py --run B            # ConvNeXt-T ImageNet-22k, fretboard strips
    python scripts/35c_train_v2n.py --run C            # S2S ResNet-18 from scratch, fretboard strips
    python scripts/35c_train_v2n.py --run A --sanity-record <record> --max-steps 300   # overfit one record
    python scripts/35c_train_v2n.py --run A --check-load                               # piano weights only

Config = V2N's configs/oofv_1s (commit a6408c4, $V2N_REPO) + configs/model/v2n_guitar/base.yaml + the run preset
below. V2N's train.py has no "initialise from other weights" option (only resume), so for the piano init its
build_model is wrapped: the model is built as usual, then the released PianoVAM weights are loaded into it
(offset/velocity heads dropped; every other key must load, checked). The V2N repo stays unmodified.

Writes outputs/runs/<date>_cp17_v2n_<run>[_<suffix>]/: config.yaml (merged), HF checkpoints, trainer_state.json,
best.safetensors (best val note onset F1, keys "transcriber.*" like the released checkpoint), log.csv (train/val
loss curves), eval_metrics.jsonl (val note / frame metrics of V2N's EvalCallback at every eval).
"""

import argparse
import csv
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2N_REPO = Path(os.environ.get("V2N_REPO", Path.home() / "external" / "V2N"))
PIANO_CKPT = Path.home() / ".cache" / "v2n" / "v2n_pianovam.safetensors"
DATA = ROOT / "data" / "processed" / "v2n_guitar"

PRESETS = {
    "A": {"version": "fret", "init": "piano", "backbone": "s2s"},
    "D": {"version": "full", "init": "piano", "backbone": "s2s"},
    "B": {"version": "fret", "init": "imagenet", "backbone": "convnext"},
    "C": {"version": "fret", "init": "scratch", "backbone": "s2s"},
}


def load_piano_into(transcriber) -> dict:
    """Released PianoVAM weights -> our onset/frame transcriber. Returns counts; raises if a key does not fit."""
    from safetensors.torch import load_file

    sd = load_file(str(PIANO_CKPT))
    sd = {k.removeprefix("transcriber."): v for k, v in sd.items()}
    dropped = sorted({k.split(".")[0] for k in sd if k.startswith(("offset_head.", "velocity_head."))})
    sd = {k: v for k, v in sd.items() if not k.startswith(("offset_head.", "velocity_head."))}
    res = transcriber.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"piano init does not fit: missing {res.missing_keys}, unexpected {res.unexpected_keys}")
    n_model = len(transcriber.state_dict())
    return {"loaded": len(sd), "model_keys": n_model, "dropped_heads": dropped}


def build_cfg(args, run_dir: Path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(V2N_REPO / "configs" / "oofv_1s"), version_base=None):
        cfg = compose(config_name="config")
    ours = OmegaConf.load(ROOT / "configs" / "model" / "v2n_guitar" / "base.yaml")
    cfg = OmegaConf.merge(cfg, ours)
    # replace (not merge) dicts whose V2N keys must disappear
    OmegaConf.update(cfg, "train.loss", ours.train.loss, merge=False)
    preset = PRESETS[args.run]
    if preset["backbone"] == "convnext":
        OmegaConf.update(cfg, "model.feature_extractor", {
            "_target_": "model.module.VisualFeatureExtractor", "backbone_name": "convnext_tiny.fb_in22k",
            "pretrained": True, "grad_sampling": 0.2}, merge=False)
    version = args.version or preset["version"]
    cfg.name = run_dir.name
    cfg.output_dir = str(run_dir)
    cfg.data_dir = str(args.data_root / version)
    cfg.eval.gt_dir = str(args.data_root / "MIDI")
    if args.sanity_record:
        cfg.data_dir = str(run_dir / "data")
        OmegaConf.update(cfg, "dataset.train.transform", {"_target_": "data.augment.IdentityClipTransform"},
                         merge=False)
        cfg.train.trainer.eval_steps = cfg.train.trainer.save_steps = 100
    if args.max_steps:
        cfg.train.trainer.max_steps = args.max_steps
    if args.batch_size:
        cfg.dataset.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.dataset.dataloader.num_workers = args.num_workers
    if args.no_compile:
        cfg.train.trainer.torch_compile = False
    return cfg, preset, version


def write_log(run_dir: Path) -> None:
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        return
    state = json.loads(states[-1].read_text())
    shutil.copy(states[-1], run_dir / "trainer_state.json")
    keys = sorted({k for e in state["log_history"] for k in e})
    with open(run_dir / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(state["log_history"])
    best = state.get("best_model_checkpoint")
    if best:
        src = Path(best)
        src = src if src.is_absolute() else run_dir / src.name
        shutil.copy(src / "model.safetensors", run_dir / "best.safetensors")
        print(f"best checkpoint: {src.name} (val note F1 {state.get('best_metric')}) -> best.safetensors")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", choices=sorted(PRESETS), required=True)
    p.add_argument("--version", choices=["fret", "full"], default=None, help="override the preset's strips")
    p.add_argument("--data-root", type=Path, default=DATA)
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--sanity-record", default=None,
                   help="overfit check: train and validate on this one record, no augmentation")
    p.add_argument("--check-load", action="store_true", help="only build the model and load the piano weights")
    args = p.parse_args()
    args.data_root = args.data_root.resolve()

    name = args.run_name or f"{datetime.date.today()}_cp17_v2n_{args.run}" + ("_sanity" if args.sanity_record else "")
    run_dir = ROOT / "outputs" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(V2N_REPO))
    os.chdir(V2N_REPO)  # V2N resolves some paths relative to its repo
    from omegaconf import OmegaConf

    import train as v2n_train

    cfg, preset, version = build_cfg(args, run_dir)
    if args.sanity_record:
        sys.path.insert(0, str(ROOT / "src"))
        from guitar2midi.baselines.v2n_data import H5_NAME, build_master

        src = args.data_root / version / "processed_v2"
        split = next(s for s in ("train", "valid") if _has_record(src / s / H5_NAME, args.sanity_record))
        for s in ("train", "valid"):
            build_master({args.sanity_record: src / split / H5_NAME}, run_dir / "data" / "processed_v2" / s / H5_NAME)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    (run_dir / "run_info.json").write_text(json.dumps({**preset, "version": version, "run": args.run,
                                                       "sanity_record": args.sanity_record}, indent=1))

    original_build = v2n_train.build_model

    def build_model(c):
        transcriber, loss_fn, wrapped = original_build(c)
        n_params = sum(q.numel() for q in transcriber.parameters())
        info = {"params_M": round(n_params / 1e6, 2), "init": preset["init"]}
        if preset["init"] == "piano":
            info.update(load_piano_into(transcriber))
        print(f"model: {type(transcriber).__name__}, {info}", flush=True)
        (run_dir / "init.json").write_text(json.dumps(info, indent=1))
        return transcriber, loss_fn, wrapped

    v2n_train.build_model = build_model

    # V2N's EvalCallback adds note F1 etc. to the eval metrics after the trainer has logged them and sends them
    # only to wandb (off here): a last callback writes them to eval_metrics.jsonl.
    from transformers import TrainerCallback

    class MetricsSink(TrainerCallback):
        def on_evaluate(self, a, state, control, metrics=None, **kw):
            if metrics:
                with open(run_dir / "eval_metrics.jsonl", "a") as f:
                    f.write(json.dumps({"step": state.global_step, **metrics}) + "\n")
                print({k: round(v, 4) for k, v in metrics.items() if k.startswith(("eval_note/", "eval_frame/"))},
                      flush=True)

    original_callbacks = v2n_train.build_callbacks
    v2n_train.build_callbacks = lambda *a, **kw: [*original_callbacks(*a, **kw), MetricsSink()]
    if args.check_load:
        build_model(cfg)
        return
    v2n_train.train(cfg)
    write_log(run_dir)


def _has_record(h5: Path, record: str) -> bool:
    import h5py

    if not h5.exists():
        return False
    with h5py.File(h5, "r") as f:
        return record in f


if __name__ == "__main__":
    main()
