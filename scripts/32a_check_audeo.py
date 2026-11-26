"""CP14 positive control: our Audeo wrapper reproduces Audeo on its own piano data (env guitar2midi).

    python scripts/32a_check_audeo.py

1. Stage 1 (Video2RollNet): the 5 example frames shipped in Audeo_dataset/input_images/testing (frames 248-252 of
   "Bach Prelude and Fugue No.1 B2") -> logits of frames 248, 249, 250 (Audeo's edge rule makes these three
   computable from 5 frames) vs Audeo's own saved logits in estimate_Roll/testing/.../248-298.npz.
2. Stage 2 (Roll2MidiNet): Audeo's saved stage-1 logits of the 3 test pieces -> our ``refine`` -> frame-level
   precision / recall / F1 against Audeo's pseudo-GT roll (midi/testing, keys 15-65), next to the stage-1 roll
   (sigmoid >= 0.4). Roll2Midi should raise F1 over the raw roll, as in the paper.
"""

import glob
from pathlib import Path

import numpy as np
from PIL import Image

from guitar2midi.baselines import audeo

PIECE = "Bach Prelude and Fugue No.{} B2"


def chunks(folder: Path, key: str) -> np.ndarray:
    files = sorted(glob.glob(str(folder / "*.npz")), key=lambda f: int(Path(f).stem.split("-")[0]))
    return np.concatenate([np.load(f)[key] for f in files])


def prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = float((pred & gt).sum())
    p, r = tp / max(pred.sum(), 1), tp / max(gt.sum(), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> None:
    d = audeo.AUDEO_DATA
    model = audeo.AudeoTranscriber()
    print(f"Audeo weights from {d}, device {model.device}")

    # 1. stage 1 on the shipped example frames
    folder = d / "input_images" / "testing" / PIECE.format(1)
    frames = np.stack([np.asarray(Image.open(folder / f"frame{i}.jpg").convert("L")) for i in range(248, 253)])
    ours = model.roll_logits(frames)[:3]
    theirs = np.load(d / "estimate_Roll" / "testing" / PIECE.format(1) / "248-298.npz")["logit"][:3,
                                                                                               audeo.MIN_KEY:audeo.MAX_KEY + 1]
    diff = np.abs(ours - theirs)
    same = ((ours > 0) == (theirs > 0)).mean()
    print(f"stage 1, frames 248-250: max |logit diff| {diff.max():.4f}, mean {diff.mean():.4f}; "
          f"same on/off decision {same:.1%}; logit range ours {ours.min():.1f}..{ours.max():.1f}")

    # 2. stage 2 on Audeo's saved stage-1 logits of the 3 test pieces
    print("\nframe-level vs Audeo's pseudo-GT (keys 15-65), per test piece:")
    print(f"{'piece':32s} {'frames':>6s}  {'roll P/R/F1':>18s}  {'Roll2Midi P/R/F1':>18s}")
    for i in range(1, 4):
        p = PIECE.format(i)
        logits = chunks(d / "estimate_Roll" / "testing" / p, "logit")[:, audeo.MIN_KEY:audeo.MAX_KEY + 1]
        gt = chunks(d / "midi" / "testing" / p, "midi")[:, audeo.MIN_KEY:audeo.MAX_KEY + 1] > 0
        n = min(len(logits), len(gt))
        roll = (1 / (1 + np.exp(-logits[:n])) >= audeo.ROLL_THRESHOLD)
        midi = model.refine(logits[:n].astype(np.float32)).astype(bool)
        r1, r2 = prf(roll, gt[:n]), prf(midi, gt[:n])
        print(f"{p:32s} {n:6d}  {r1[0]:.2f}/{r1[1]:.2f}/{r1[2]:.2f}      {r2[0]:.2f}/{r2[1]:.2f}/{r2[2]:.2f}")


if __name__ == "__main__":
    main()
