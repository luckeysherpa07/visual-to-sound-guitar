"""CP6 positive control: does our V2N wrapper reproduce V2N on piano? (env ``v2n``)

    python scripts/30g_check_v2n_piano.py

Runs guitar2midi.baselines.v2n on one PianoVAM test video (2024-02-17_21-44-37, smallest test file, from
HF PianoVAM/PianoVAM_v1 into data/raw/pianovam/), preprocessed with V2N's own function
(data.preprocess_video: corner warp, 800x144, bottom_margin 100, grayscale, 25 fps, JPEG q95 round trip as
in their HDF5), and compares the result with
  (a) V2N's shipped prediction predicted_midi/pianovam_test/<id>.mid  -> should agree ~100 %
  (b) the PianoVAM ground-truth MIDI                                   -> should be near the paper (~95 % onset F1)
If both hold, a silent output on guitar strips is V2N's behaviour, not a bug in our input path.
Writes outputs/runs/<date>_cp6_v2n_piano_check/{result.json, strip.png}.
"""

import json
from datetime import date
from pathlib import Path

import cv2
import mir_eval
import numpy as np
import pretty_midi

from guitar2midi.baselines import v2n

REC = "2024-02-17_21-44-37"
ROOT = Path("data/raw/pianovam")


def notes(path: Path):
    pm = pretty_midi.PrettyMIDI(str(path))
    ns = [n for i in pm.instruments for n in i.notes]
    iv = np.array([[n.start, max(n.end, n.start + 1e-3)] for n in ns]).reshape(-1, 2)
    return iv, mir_eval.util.midi_to_hz(np.array([n.pitch for n in ns], float))


def f1(ref, est, tol=0.05):
    return mir_eval.transcription.precision_recall_f1_overlap(*ref, *est, onset_tolerance=tol, offset_ratio=None)[2]


def main() -> None:
    v2n._import_v2n()
    from data.preprocess_video import process_video_opencv_grab_retrieve

    meta = json.load(open(ROOT / "metadata.json"))
    rec = next(r for r in (meta.values() if isinstance(meta, dict) else meta) if r["record_time"] == REC)
    jpgs, dur, _ = process_video_opencv_grab_retrieve(str(ROOT / "Video" / f"{REC}.mp4"), rec, 800, 144, 25,
                                                       bottom_margin=100, grayscale=True, jpeg_quality=95)
    strips = np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in jpgs])
    print(f"{REC}: {len(strips)} frames ({dur:.0f} s)")

    model = v2n.V2NTranscriber()
    pred = model.decode(model.probs(strips))
    run = Path("outputs/runs") / f"{date.today()}_cp6_v2n_piano_check"
    run.mkdir(parents=True, exist_ok=True)
    v2n.write_midi(pred, run / f"{REC}_ours.mid")
    cv2.imwrite(str(run / "strip.png"), strips[len(strips) // 2])

    ours = notes(run / f"{REC}_ours.mid")
    shipped = notes(v2n.V2N_REPO / "predicted_midi" / "pianovam_test" / f"{REC}.mid")
    gt = notes(ROOT / "MIDI" / f"{REC}.mid")
    res = {"record": REC, "n_ours": len(ours[0]), "n_shipped": len(shipped[0]), "n_gt": len(gt[0]),
           "ours_vs_shipped_onset_f1_50": f1(shipped, ours), "ours_vs_gt_onset_f1_50": f1(gt, ours),
           "shipped_vs_gt_onset_f1_50": f1(gt, shipped), "ours_vs_gt_onset_f1_100": f1(gt, ours, 0.1)}
    json.dump(res, open(run / "result.json", "w"), indent=2)
    for k, v in res.items():
        print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")


if __name__ == "__main__":
    main()
