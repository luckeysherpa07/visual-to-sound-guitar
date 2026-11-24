"""CP8 spot check of the reference transcription on magcil (no note labels exist).

    python scripts/31b_spot_check_magcil.py                 # env guitar2midi

1. Per clip: notes, notes/s, pitch range, share of very short notes (< 50 ms), notes outside the guitar range
   (E2..E6 = MIDI 40..88; bends/harmonics can go above the fretted range).
2. Score check: the exercises' MuseScore files give the pitch classes each exercise uses. The recordings start
   from different positions (transposed), and recording -> score file is not documented for every class, so the
   best-fitting exercise of the class and the best transposition are used. Reported: share of transcribed notes
   whose pitch class is in that set, vs chance (= set size / 12). Lenient; a sanity check, not a metric.
3. Spectrograms with the transcribed notes overlaid (white) for 3 clips (--plots).

Writes <clips-dir>/spot_check_real_audio.csv and plots/spec_<clip>.png.
"""

import argparse
import csv
import re
import zipfile
from pathlib import Path

import numpy as np
import pretty_midi

from guitar2midi.baselines.audio_plots import plot_spectrograms
from guitar2midi.data import magcil

CLIPS = Path("data/interim/baseline_clips_magcil")


def score_pitches(mscz: Path) -> np.ndarray:
    with zipfile.ZipFile(mscz) as z:
        xml = z.read(next(n for n in z.namelist() if n.endswith(".mscx"))).decode()
    return np.array([int(p) for p in re.findall(r"<pitch>(\d+)</pitch>", xml)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=CLIPS)
    p.add_argument("--system", default="real_audio")
    p.add_argument("--plots", nargs="*", default=["alternate picking", "legato", "bend"],
                   help="techniques to plot (first clip of each)")
    args = p.parse_args()

    with open(args.clips_dir / "clips.csv") as f:
        clips = list(csv.DictReader(f))
    scores = {k: {m.stem: set(score_pitches(m) % 12) for m in sorted((magcil.MAGCIL_ROOT / "musescore" / f"Class_{k}").glob("*.mscz"))}
              for k in magcil.TECHNIQUES}
    rows = []
    for c in clips:
        m = pretty_midi.PrettyMIDI(str(args.clips_dir / "transcriptions" / args.system / f"{c['clip_id']}.mid"))
        notes = [n for i in m.instruments for n in i.notes]
        pitches = np.array([n.pitch for n in notes])
        length = float(c["length_s"])
        best = (0.0, "", 0, 0)
        for name, pcs in scores[int(c["class"])].items():
            for shift in range(12):
                share = float(np.isin((pitches - shift) % 12, list(pcs)).mean()) if len(pitches) else 0.0
                if share > best[0]:
                    best = (share, name, shift, len(pcs))
        rows.append({"clip_id": c["clip_id"], "technique": c["technique"], "guitar": c["guitar"],
                     "n_notes": len(notes), "notes_per_s": round(len(notes) / length, 2),
                     "pitch_min": int(pitches.min()) if len(notes) else "", "pitch_max": int(pitches.max()) if len(notes) else "",
                     "short_lt50ms": round(float(np.mean([n.end - n.start < 0.05 for n in notes])), 3) if notes else "",
                     "outside_40_88": int(((pitches < 40) | (pitches > 88)).sum()),
                     "score_pc_share": round(best[0], 3), "score_chance": round(best[3] / 12, 3),
                     "score_file": best[1], "score_shift": best[2]})
        r = rows[-1]
        print(f"{c['clip_id']:40s} {r['technique']:17s} notes {r['n_notes']:3d} ({r['notes_per_s']:4.1f}/s) "
              f"pitch {r['pitch_min']}-{r['pitch_max']}  short {r['short_lt50ms']}  "
              f"score pc {r['score_pc_share']:.0%} (chance {r['score_chance']:.0%})")
    with open(args.clips_dir / f"spot_check_{args.system}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for tech in args.plots:
        c = next(c for c in clips if c["technique"] == tech)
        m = pretty_midi.PrettyMIDI(str(args.clips_dir / "transcriptions" / args.system / f"{c['clip_id']}.mid"))
        notes = np.array([(n.start, n.end, n.pitch) for i in m.instruments for n in i.notes])
        plot_spectrograms({"real audio": args.clips_dir / "audio" / f"{c['clip_id']}.wav"},
                          args.clips_dir / "plots" / f"spec_{c['clip_id']}.png",
                          f"{c['clip_id']} ({tech}), reference transcription", notes if len(notes) else None,
                          duration=float(c["length_s"]))

    share = np.array([r["score_pc_share"] for r in rows])
    chance = np.array([r["score_chance"] for r in rows])
    print("\n== CP8 spot check ==")
    print(f"notes/clip:       median {int(np.median([r['n_notes'] for r in rows]))}, "
          f"notes/s median {np.median([r['notes_per_s'] for r in rows]):.1f}")
    print(f"outside E2-E6:    {sum(r['outside_40_88'] for r in rows)} notes")
    print(f"score pitch class share: median {np.median(share):.0%} vs chance median {np.median(chance):.0%}")
    print(f"outputs: {args.clips_dir}/spot_check_{args.system}.csv, plots/spec_*.png")


if __name__ == "__main__":
    main()
