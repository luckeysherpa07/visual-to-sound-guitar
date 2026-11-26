import shutil
from pathlib import Path

import torch


def save_checkpoint(state, is_best: bool = False, folder: str = ".", filename: str = "checkpoint.pth.tar"):
    path = Path(folder) / filename
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, Path(folder) / "model_best.pth.tar")
