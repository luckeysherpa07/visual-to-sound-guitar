"""CP10 re-check: input statistics of our guitar strips vs V2N's piano strips (env ``v2n``).

    python scripts/30g_strip_stats.py

Piano: the PianoVAM test video of the CP6 positive control, preprocessed with V2N's own function (as in
30g_check_v2n_piano.py). Guitar: valid frames of all strips per dataset. Per frame: mean and std of the pixel
values (0-255) and edge density (share of pixels with Sobel magnitude > 50); reported as median over frames.
Also saves a side-by-side image of one piano strip and one strip per dataset.
Writes outputs/runs/<date>_cp10_strip_stats/{stats.md, strips.png}.
"""

import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from guitar2midi.baselines import v2n

DATASETS = {"gaps": Path("data/interim/baseline_clips"), "guifir": Path("data/interim/baseline_clips_guifir"),
            "magcil": Path("data/interim/baseline_clips_magcil")}
REC, ROOT = "2024-02-17_21-44-37", Path("data/raw/pianovam")


def frame_stats(strips: np.ndarray) -> np.ndarray:
    out = []
    for f in strips:
        g = np.hypot(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))
        out.append((f.mean(), f.std(), (g > 50).mean()))
    return np.array(out)


def main() -> None:
    v2n._import_v2n()
    from data.preprocess_video import process_video_opencv_grab_retrieve

    meta = json.load(open(ROOT / "metadata.json"))
    rec = next(r for r in (meta.values() if isinstance(meta, dict) else meta) if r["record_time"] == REC)
    jpgs, _, _ = process_video_opencv_grab_retrieve(str(ROOT / "Video" / f"{REC}.mp4"), rec, 800, 144, 25,
                                                    bottom_margin=100, grayscale=True, jpeg_quality=95)
    piano = np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in jpgs])
    stats = {"piano (PianoVAM, V2N preprocessing)": frame_stats(piano[::5])}
    examples = [piano[len(piano) // 2]]
    for name, d in DATASETS.items():
        frames = []
        for f in sorted((d / "strips").glob("*.npy")):
            if f.stem.endswith(("_valid", "_corners")):
                continue
            s = np.load(f)
            valid = np.load(f.with_name(f"{f.stem}_valid.npy")).astype(bool)[: len(s)]
            frames.append(s[valid][::5])
        frames = np.concatenate(frames)
        stats[name] = frame_stats(frames)
        examples.append(frames[len(frames) // 2])

    lines = ["| strips | frames | mean | std | edge density |", "|---|---|---|---|---|"]
    for name, s in stats.items():
        med = np.median(s, 0)
        lines.append(f"| {name} | {len(s)} | {med[0]:.0f} | {med[1]:.0f} | {med[2]:.3f} |")
    out = Path("outputs/runs") / f"{date.today().isoformat()}_cp10_strip_stats"
    out.mkdir(parents=True, exist_ok=True)
    md = ("Median over frames (every 5th valid frame). Pixel values 0-255; edge density = share of pixels with Sobel "
          "gradient magnitude > 50.\n\n" + "\n".join(lines) + "\n")
    (out / "stats.md").write_text(md)
    labels = ["piano"] + list(DATASETS)
    rows = [cv2.putText(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR), l, (5, 20), 0, 0.6, (0, 0, 255), 2)
            for e, l in zip(examples, labels)]
    cv2.imwrite(str(out / "strips.png"), np.vstack(rows))
    print(md + f"\nwritten to {out}")


if __name__ == "__main__":
    main()
