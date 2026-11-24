"""CP10 re-check of the V2N zero-shot result (silence on GAPS in CP6): is there any signal in V2N's output on
guitar strips, for any strip orientation / layout, on any dataset? (env ``v2n``)

    python scripts/30g_v2n_recheck.py --run-name cp10_v2n_recheck

Needs the probabilities written by 30g_run_v2n.py for each variant (<clips-dir>/v2n_probs[/<system>]/).
Per dataset (GAPS, GuiFiR-Vid, magcil) and variant (v2n.FLIPS x v2n.LAYOUTS x norm) it reports:
  - max onset prob (median / max over clips): does anything get near the 0.5 decoding threshold?
  - timing signal, threshold-free: Pearson r between V2N's per-frame onset activity (max over pitches) and the
    reference onset curve (smoothed, 25 fps), for the matching clip vs a clip of another piece/song/technique
    (same pairing as 30e --shuffled). Paired Wilcoxon over clips; "better" = share of clips matched > shuffled.
  - pitch signal, threshold-free: same, but per pitch (onset probs vs reference onset roll, all 88 pitches
    flattened).
  - note metrics with lower decoding thresholds (onset = frame threshold = 0.5, 0.2, 0.1, 0.05), diagnostic
    only (never for the results table): notes per clip, onset F1@50 ms, timing F1.
Only valid frames (strips/<clip>_valid.npy) are used; reference = GAPS MIDI or the real-audio transcription.
Writes outputs/runs/<date>_<run-name>/{recheck.csv, summary.md}.
"""

import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from guitar2midi.baselines import v2n
from guitar2midi.evaluation import note_metrics as M

DATASETS = {"gaps": Path("data/interim/baseline_clips"), "guifir": Path("data/interim/baseline_clips_guifir"),
            "magcil": Path("data/interim/baseline_clips_magcil")}
THRESHOLDS = (0.5, 0.2, 0.1, 0.05)


def system_name(flip: str, layout: str, norm: bool = False) -> str:
    if (flip, layout, norm) == ("none", "mid", False):
        return "v2n_zeroshot"
    return f"v2n_zeroshot_{flip}_{layout}" + ("_norm" if norm else "")


def load_clips(d: Path) -> list[dict]:
    with open(d / "clips.csv") as f:
        clips = [r for r in csv.DictReader(f) if r["keep"] == "1" and (d / "strips" / f"{r['clip_id']}_valid.npy").exists()]
    group = next(k for k in ("piece_id", "song", "technique", "clip_id") if k in clips[0])
    for c in clips:
        c["_group"] = c[group]
    return clips


def partner(clips: list[dict], i: int) -> dict:
    for k in range(3, len(clips) + 3):
        o = clips[(i + k) % len(clips)]
        if o["_group"] != clips[i]["_group"]:
            return o
    raise ValueError


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def onset_roll(notes: np.ndarray, n: int) -> np.ndarray:
    roll = np.zeros((n, 88))
    for on, _, p in notes:
        i, k = int(round(on * M.FPS)), int(p) - v2n.MIN_MIDI
        if 0 <= i < n and 0 <= k < 88:
            roll[max(i - 1, 0):i + 2, k] = 1  # +-1 frame
    return roll


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="cp10_v2n_recheck")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    args = p.parse_args()

    import torch
    _, NoteDecoder = v2n._import_v2n()
    decoders = {t: NoteDecoder(onset_threshold=t, frame_threshold=t, fps=v2n.FPS, prefer_onset_predictions=True,
                               prefer_offset_predictions=True) for t in THRESHOLDS}

    rows = []
    for ds in args.datasets:
        d = DATASETS[ds]
        clips = load_clips(d)
        ref_dir = d / "midi" if (d / "midi").is_dir() else d / "transcriptions" / "real_audio"
        refs = {c["clip_id"]: M.load_notes(ref_dir / f"{c['clip_id']}.mid") for c in clips}
        valid = {c["clip_id"]: np.load(d / "strips" / f"{c['clip_id']}_valid.npy") for c in clips}
        for flip, layout, norm in [(f, l, n) for n in (False, True) for f in v2n.FLIPS for l in v2n.LAYOUTS]:
            system = system_name(flip, layout, norm)
            pdir = d / "v2n_probs" if system == "v2n_zeroshot" else d / "v2n_probs" / system
            if not all((pdir / f"{c['clip_id']}.npz").exists() for c in clips):
                print(f"{ds} {system}: probabilities missing, skipped")
                continue
            probs = {c["clip_id"]: {k: v.astype(np.float32) for k, v in np.load(pdir / f"{c['clip_id']}.npz").items()}
                     for c in clips}
            tm, ts, pm, ps, max_on = [], [], [], [], []
            dec = {t: [] for t in THRESHOLDS}
            for i, c in enumerate(clips):
                cid, other = c["clip_id"], partner(clips, i)["clip_id"]
                on = probs[cid]["onset_probs"]
                T = len(on)
                v = valid[cid][:T].astype(bool)
                act = on.max(1)
                dur = T / M.FPS
                tm.append(corr(act[v], M.onset_curve(refs[cid], dur)[:T][v]))
                ts.append(corr(act[v], M.onset_curve(refs[other], dur)[:T][v]))
                pm.append(corr(on[v].ravel(), onset_roll(refs[cid], T)[v].ravel()))
                ps.append(corr(on[v].ravel(), onset_roll(refs[other], T)[v].ravel()))
                max_on.append(float(on[v].max()) if v.any() else 0.0)
                for t in THRESHOLDS:
                    notes = decoders[t]({k: torch.from_numpy(x) for k, x in probs[cid].items()})
                    est = (np.zeros((0, 3)) if len(notes) == 0 else
                           np.column_stack([notes.intervals.astype(float) / v2n.FPS,
                                            np.asarray(notes.pitches) + v2n.MIN_MIDI]))
                    m = M.evaluate(refs[cid], est, dur, valid=valid[cid])
                    dec[t].append((m["n_est"], m["onset_f50"], m["timing_f50"]))
            tm, ts, pm, ps = map(np.array, (tm, ts, pm, ps))
            row = {"dataset": ds, "system": system, "flip": flip, "layout": layout, "norm": int(norm), "clips": len(clips),
                   "max_onset_median": round(float(np.median(max_on)), 4), "max_onset_max": round(max(max_on), 4),
                   "timing_r_matched": round(float(tm.mean()), 4), "timing_r_shuffled": round(float(ts.mean()), 4),
                   "timing_better": round(float(np.mean(tm > ts)), 3),
                   "timing_p": float(wilcoxon(tm, ts).pvalue) if np.any(tm != ts) else 1.0,
                   "pitch_r_matched": round(float(pm.mean()), 4), "pitch_r_shuffled": round(float(ps.mean()), 4),
                   "pitch_better": round(float(np.mean(pm > ps)), 3),
                   "pitch_p": float(wilcoxon(pm, ps).pvalue) if np.any(pm != ps) else 1.0}
            for t in THRESHOLDS:
                a = np.array(dec[t])
                row.update({f"notes@{t}": round(float(a[:, 0].mean()), 1), f"onsetF1@{t}": round(float(a[:, 1].mean()), 4),
                            f"timingF1@{t}": round(float(a[:, 2].mean()), 4)})
            rows.append(row)
            print(f"{ds:6s} {system:22s} max on {row['max_onset_median']:.3f}/{row['max_onset_max']:.3f}  "
                  f"timing r {row['timing_r_matched']:+.3f} vs {row['timing_r_shuffled']:+.3f} (p {row['timing_p']:.2g})  "
                  f"pitch r {row['pitch_r_matched']:+.3f} vs {row['pitch_r_shuffled']:+.3f} (p {row['pitch_p']:.2g})  "
                  + "  ".join(f"@{t}: {row[f'notes@{t}']:.0f}n F1 {100 * row[f'onsetF1@{t}']:.1f}" for t in THRESHOLDS),
                  flush=True)

    out = Path("outputs/runs") / f"{date.today().isoformat()}_{args.run_name}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "recheck.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| dataset | flip | layout | norm | max onset p (median / max) | timing r matched / shuffled (p) | "
             "pitch r matched / shuffled (p) | notes/clip @0.5 / 0.1 / 0.05 | onset F1 @0.5 / 0.1 / 0.05 | timing F1 @0.05 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['flip']} | {r['layout']} | {'yes' if r['norm'] else 'no'} | {r['max_onset_median']:.3f} / {r['max_onset_max']:.3f} | "
                     f"{r['timing_r_matched']:+.3f} / {r['timing_r_shuffled']:+.3f} ({r['timing_p']:.2g}) | "
                     f"{r['pitch_r_matched']:+.3f} / {r['pitch_r_shuffled']:+.3f} ({r['pitch_p']:.2g}) | "
                     f"{r['notes@0.5']:.1f} / {r['notes@0.1']:.1f} / {r['notes@0.05']:.1f} | "
                     f"{100 * r['onsetF1@0.5']:.1f} / {100 * r['onsetF1@0.1']:.1f} / {100 * r['onsetF1@0.05']:.1f} | "
                     f"{100 * r['timingF1@0.05']:.1f} |")
    md = ("V2N zero-shot re-check. flip: h = nut right, v = low E on top; layout: mid = neck in the middle half "
          "(our default), top = neck in the top 2/3 + margin below (like V2N's piano crop); norm = each frame "
          "rescaled to the piano strips' mean/std. r = Pearson correlation "
          "of V2N onset activity with the reference onset curve (timing) / onset roll (pitch), matched clip vs a clip "
          "of another piece; p = paired Wilcoxon. Lower thresholds = diagnostic only.\n\n" + "\n".join(lines) + "\n")
    (out / "summary.md").write_text(md)
    print("\n" + md + f"\nwritten to {out}")


if __name__ == "__main__":
    main()
