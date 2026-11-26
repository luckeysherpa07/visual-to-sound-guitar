from pathlib import Path

from pyhocon import HOCONConverter

from ._args import args
from .config import config

experiment_path = None
if args.experiment:
    _p = Path(args.experiment)
    _p.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (_p / "config.conf").write_text(HOCONConverter.to_hocon(config))
    experiment_path = str(_p)
