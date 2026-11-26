import logging as _logging
import sys
from pathlib import Path

from .environment import experiment_path

logger = _logging.getLogger("foley_music")
logger.setLevel(_logging.INFO)
if not logger.handlers:
    fmt = _logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    handlers = [_logging.StreamHandler(sys.stdout)]
    if experiment_path:
        handlers.append(_logging.FileHandler(Path(experiment_path) / "log.txt"))
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)
