Stand-in for `torchpie` v0.2 (SunDoge), which Foley Music's `train.py` / `core/` import. The original package was
hosted on git.dev.tencent.com (offline) and is not on PyPI or GitHub (checked 2026-11-26). Only the five helpers
Foley Music uses are provided, with the behaviour Foley Music relies on:

- `torchpie.config.config`: pyhocon ConfigTree of the file given with `-c/--config`
- `torchpie.environment.experiment_path`: the folder given with `-e/--experiment` (created; the merged config is
  saved there as `config.conf`, which `test_*.py` reads next to the checkpoint)
- `torchpie.logging.logger`: a `logging.Logger` writing to stdout and `<experiment>/log.txt`
- `torchpie.meters.AverageMeter(name)`: running average, printed as `name val (avg)`
- `torchpie.utils.checkpoint.save_checkpoint(state, is_best, folder)`: `checkpoint.pth.tar`, copied to
  `model_best.pth.tar` when `is_best`

Also `foley_torch2_patch.py` (applied at start-up by `sitecustomize.py`): restores the torch-1.x `nn.Transformer.forward` (torch >= 2.1 passes `is_causal` arguments
that Foley Music's custom encoder/decoder do not accept). Python imports it automatically at start-up.

Put this folder (not the `torchpie` folder) on `PYTHONPATH` when running Foley Music; never imported by our package.
