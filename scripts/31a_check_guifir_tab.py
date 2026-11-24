"""CP7 tab check: how often does a transcription contain the GuiFiR tab pitches?

    python scripts/31a_check_guifir_tab.py                          # reference = real_audio transcription
    python scripts/31a_check_guifir_tab.py --system mmaudio_noprompt  # any system later (CP9/CP10)

GuiFiR test tabs are manual and mark picking moments (string + fret + capo). For every tab pitch:
    onset     a note of that pitch starts within +-100 ms
    sounding  a note of that pitch is active at the tab frame
    chroma    like onset, pitch class only
Recall only: the tab is sparse, so extra notes cannot be penalised. String order and capo handling are not
documented; all readings in ``clips_guifir.TAB_VARIANTS`` are scored (printed); CP7 found frame_tab[0] = high E
+ capo, which is the default for every system (same reading for all, so systems are comparable).

Writes <clips-dir>/tab_check_<system>.csv (per clip, best variant) and prints the summary.
Also draws <clips-dir>/plots/tab_<clip>.png (transcription piano roll + tab pitches) for --plots clips.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pretty_midi

from guitar2midi.baselines import clips_guifir as G

CLIPS = Path("data/interim/baseline_clips_guifir")


def plot(midi, events, variant, dst, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    for n in (n for i in midi.instruments for n in i.notes):
        ax.barh(n.pitch, n.end - n.start, left=n.start, height=0.8, color="tab:blue", alpha=0.6)
    for ev in events:
        for p in G.tab_pitches(ev["frame_tab"], ev["capo"], variant):
            ax.plot(ev["t"], p, "rx", ms=7, mew=2)
    ax.set_xlabel("time in clip (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title(f"{title}: blue = transcription, red x = tab ({variant})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=100)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", type=Path, default=CLIPS)
    p.add_argument("--system", default="real_audio")
    p.add_argument("--variant", default="high_first+capo",
                   help="tab reading (CP7: frame_tab[0] = high E, + capo); 'best' = best on this system")
    p.add_argument("--plots", type=int, default=3)
    args = p.parse_args()

    tabs = {f.stem: json.loads(f.read_text()) for f in sorted((args.clips_dir / "tab").glob("*.json"))}
    tdir = args.clips_dir / "transcriptions" / args.system
    midis = {c: pretty_midi.PrettyMIDI(str(tdir / f"{c}.mid")) for c in tabs if (tdir / f"{c}.mid").exists()}
    print(f"{len(midis)}/{len(tabs)} clips with a transcription in {tdir}")

    totals = {}
    for v in G.TAB_VARIANTS:
        r = [G.tab_recall(midis[c], tabs[c]["events"], v) for c in midis]
        n = sum(x["n_tab_pitches"] for x in r)
        totals[v] = {k: sum(x[k] for x in r) / max(n, 1) for k in ("onset", "sounding", "chroma")} | {"n": n}
    print("\ntab reading        onset  sounding  chroma   (share of tab pitches found)")
    for v, t in totals.items():
        print(f"  {v:16s} {t['onset']:6.1%} {t['sounding']:8.1%} {t['chroma']:7.1%}")
    variant = max(totals, key=lambda v: totals[v]["onset"]) if args.variant == "best" else args.variant
    print(f"-> using '{variant}'")

    rows = []
    for c, m in midis.items():
        r = G.tab_recall(m, tabs[c]["events"], variant)
        n = r["n_tab_pitches"]
        rows.append({"clip_id": c, "variant": variant, "n_tab_pitches": n,
                     **{k: round(r[k] / max(n, 1), 3) for k in ("onset", "sounding", "chroma")},
                     "n_notes_transcribed": sum(len(i.notes) for i in m.instruments)})
    with open(args.clips_dir / f"tab_check_{args.system}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    onset = np.array([r["onset"] for r in rows])
    print(f"\nper clip onset recall: median {np.median(onset):.1%}, min {onset.min():.1%} "
          f"({min(rows, key=lambda r: r['onset'])['clip_id']}), clips < 50 %: {(onset < 0.5).sum()}")
    for r in sorted(rows, key=lambda r: r["onset"])[:5]:
        print(f"  low: {r['clip_id']:32s} onset {r['onset']:.0%}  sounding {r['sounding']:.0%}  "
              f"chroma {r['chroma']:.0%}  ({r['n_tab_pitches']} tab pitches)")
    ordered = sorted(rows, key=lambda r: r["onset"])
    picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]][: args.plots]
    for r in picks:
        plot(midis[r["clip_id"]], tabs[r["clip_id"]]["events"], variant,
             args.clips_dir / "plots" / f"tab_{r['clip_id']}.png", f"{r['clip_id']} onset recall {r['onset']:.0%}")
    print(f"outputs: {args.clips_dir}/tab_check_{args.system}.csv, plots/tab_*.png")


if __name__ == "__main__":
    main()
