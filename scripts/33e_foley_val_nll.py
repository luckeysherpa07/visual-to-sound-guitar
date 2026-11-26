"""CP15: does the retrained Foley Music use the pose, and how much do our keypoints cost? (env foleymusic)

    PYTHONNOUSERSITE=1 python scripts/33e_foley_val_nll.py
    PYTHONNOUSERSITE=1 python scripts/33e_foley_val_nll.py --checkpoint checkpoint.pth.tar --tag last   # epoch 100
    PYTHONNOUSERSITE=1 python scripts/33e_foley_val_nll.py --exp outputs/runs/2026-11-26_cp15_foleymusic_urmp_violin/exp \
        --no-ours --tag violin     # same test for the URMP violin model (all its val windows, theirs vs shuffled only)

Foley Music's own validation pieces (MUSIC acoustic guitar, 6 s windows, their MIDI targets, their val data loader)
of the videos for which we have our keypoints (33d: 4 videos, first 120 s), scored with teacher forcing
(cross-entropy per MIDI event = the paper's NLL) under three pose inputs:
  theirs    their released OpenPose keypoints (the training condition)
  ours      our MMPose keypoints of the same frames (the condition on our clips)
  shuffled  their keypoints of another validation window (another video): the loss without matching video
theirs << shuffled: the model uses the pose; ours close to theirs: our keypoints are a fair substitute.
With --no-ours: all validation windows of the model's dataset, theirs vs shuffled only (e.g. URMP violin, where we
have no keypoints of our own). Writes outputs/runs/<date>_cp15_foleymusic_val_nll/{per_window,summary}[_<tag>].*
"""

import argparse
import csv
import os
from datetime import date
from pathlib import Path

import numpy as np

from guitar2midi.baselines import foley_music as FM

DATA = Path.home() / "external" / "FoleyMusic_data"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", type=Path, default=FM.EXP)
    p.add_argument("--checkpoint", default="model_best.pth.tar")
    p.add_argument("--no-ours", action="store_true", help="theirs vs shuffled on all val windows (no own keypoints)")
    p.add_argument("--tag", default="", help="suffix of the output files")
    args = p.parse_args()

    root = Path.cwd()
    exp = args.exp.resolve()
    out = root / "outputs" / "runs" / f"{date.today().isoformat()}_cp15_foleymusic_val_nll"
    out.mkdir(parents=True, exist_ok=True)
    fm = FM.FoleyMusic(exp, args.checkpoint)
    torch = fm.torch
    os.chdir(exp.parent)  # config paths (data/, resources/) are relative to the run folder
    from core.dataloaders import DataLoaderFactory

    ds = DataLoaderFactory(fm.cfg).build(split="val").dataset
    ours = {} if args.no_ours else {f.stem: np.load(f) for f in (DATA / "pose_ours").glob("*.npy")}
    idx = list(range(len(ds))) if args.no_ours else [
        i for i, s in enumerate(ds.samples) if s.vid in ours and int(s.start_time * FM.FPS) + fm.window <= len(ours[s.vid])]
    print(f"{len(ds)} val windows, {len(idx)} used ({sorted(ours) or 'theirs vs shuffled only'})")
    ce = torch.nn.CrossEntropyLoss(ignore_index=ds.PAD_IDX, reduction="sum")
    rows = []
    for j, i in enumerate(idx):
        s, item = ds.samples[i], ds[i]
        k = idx[(j + len(idx) // 2) % len(idx)]  # partner window, half the list away (another video)
        for step in range(len(idx)):  # make sure the partner is another video
            if ds.samples[k].vid != s.vid:
                break
            k = idx[(j + len(idx) // 2 + step + 1) % len(idx)]
        sf = int(s.start_time * FM.FPS)
        poses = {"theirs": item["pose"], "shuffled": ds[k]["pose"]}
        if not args.no_ours:
            poses["ours"] = torch.from_numpy(np.transpose(ours[s.vid][sf:sf + fm.window], (1, 0, 2, 3)).copy())
        mx, my = item["midi_x"][None].to(fm.device), item["midi_y"][None].to(fm.device)
        r = {"vid": s.vid, "start": s.start_time, "partner": ds.samples[k].vid,
             "n_events": int((my != ds.PAD_IDX).sum())}
        with torch.no_grad():
            for name, pose in poses.items():
                y = fm.model(pose[None].float().to(fm.device), mx, pad_idx=ds.PAD_IDX)
                r[name] = float(ce(y.reshape(-1, y.shape[-1]), my.flatten())) / max(r["n_events"], 1)
        rows.append(r)
    sfx = f"_{args.tag}" if args.tag else ""
    names = [n for n in ("theirs", "ours", "shuffled") if n in rows[0]]
    with open(out / f"per_window{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"{len(rows)} val windows (6 s) of {len({r['vid'] for r in rows})} videos; NLL per MIDI event "
             f"(teacher forcing), checkpoint {exp}/{args.checkpoint}"]
    from scipy.stats import wilcoxon
    for name in names:
        v = np.array([r[name] for r in rows])
        lines.append(f"  {name:8s}: mean {v.mean():.3f}, median {np.median(v):.3f}")
    for a, b in [x for x in (("theirs", "shuffled"), ("ours", "shuffled"), ("theirs", "ours")) if set(x) <= set(names)]:
        x, y = np.array([r[a] for r in rows]), np.array([r[b] for r in rows])
        lines.append(f"  {a} < {b}: {np.mean(x < y):.0%} of windows, Wilcoxon p = {wilcoxon(x, y).pvalue:.2g}")
    (out / f"summary{sfx}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
