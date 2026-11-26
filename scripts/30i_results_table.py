"""Baseline results table for all datasets (CP11, extended with the CP13-CP15 rows; env guitar2midi).

    python scripts/30i_results_table.py

Reads the 30e runs of each dataset (masked to the clips/frames with strips) and merges their systems:
    outputs/runs/2026-11-26_cp13_v2a_{gaps,guifir,magcil}/per_clip.csv     (V2A, V2N, controls)
    outputs/runs/2026-11-26_cp14_audeo_{gaps,guifir,magcil}/per_clip.csv   (Audeo, CP14; 32b_run_audeo.py)
    outputs/runs/2026-11-26_cp15_foleymusic_{gaps,guifir,magcil}/per_clip.csv   (Foley Music, CP15; 33c_run_foley_music.py)
created with (outputs/logs/cp13_evaluate.sh)
    python scripts/30e_evaluate_baselines.py --clips <clips-dir> --run-name cp13_v2a_<ds> --mask-dir <clips-dir>/strips \\
        --systems real_audio mmaudio_noprompt mmaudio_prompt thinksound_noprompt thinksound_prompt \\
                  foleycrafter_noprompt foleycrafter_prompt v2n_zeroshot silence \\
        --shuffled real_audio mmaudio_noprompt thinksound_noprompt thinksound_prompt foleycrafter_noprompt foleycrafter_prompt
(the CP11 runs 2026-11-24_cp11_* give identical numbers for the systems they contain)
and adds for GuiFiR-Vid the tab recall (share of the manual tab pitches found as onsets within +-100 ms,
``clips_guifir.tab_recall``, reading high E first + capo), with the same shuffled pairing as 30e.

Rows per dataset: real audio (GAPS: ceiling; pseudo-GT sets: the reference itself, 100 by construction, so only
its tab recall is shown), MMAudio / ThinkSound / FoleyCrafter without / with prompt, V2N and Audeo zero-shot
(video -> MIDI on the fretboard strips), Foley Music retrained on its MUSIC guitar data (body + hand keypoints -> MIDI), chance
(real-audio transcription of another piece) and silence.
Significance: for every system that has a ``<system>@shuffled`` run (the same model given another piece's video),
paired Wilcoxon over clips, matched vs mismatched video; * = p < 0.05 and matched better. All matched/mismatched
means are in tests.csv.

Writes thesis/tables/baseline_results.tex and outputs/runs/<date>_results_table/{table.md, table.csv, tests.csv}.
"""

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.stats import wilcoxon

from guitar2midi.baselines import clips_guifir as G

DATASETS = [  # name in table, 30e runs (merged), clips dir, label source
    ("GAPS", ("2026-11-26_cp13_v2a_gaps", "2026-11-26_cp14_audeo_gaps",
              "2026-11-26_cp15_foleymusic_gaps"), Path("data/interim/baseline_clips"), "real MIDI"),
    ("GuiFiR-Vid", ("2026-11-26_cp13_v2a_guifir", "2026-11-26_cp14_audeo_guifir",
              "2026-11-26_cp15_foleymusic_guifir"), Path("data/interim/baseline_clips_guifir"), "pseudo-GT"),
    ("magcil", ("2026-11-26_cp13_v2a_magcil", "2026-11-26_cp14_audeo_magcil",
              "2026-11-26_cp15_foleymusic_magcil"), Path("data/interim/baseline_clips_magcil"), "pseudo-GT"),
]
ROWS = [  # system in per_clip.csv, label
    ("real_audio", "Real audio"),
    ("mmaudio_noprompt", "MMAudio (video)"),
    ("mmaudio_prompt", "MMAudio (+ prompt)"),
    ("thinksound_noprompt", "ThinkSound (video)"),
    ("thinksound_prompt", "ThinkSound (+ prompt)"),
    ("foleycrafter_noprompt", "FoleyCrafter (video)"),
    ("foleycrafter_prompt", "FoleyCrafter (+ prompt)"),
    ("v2n_zeroshot", "V2N zero-shot"),
    ("audeo_zeroshot", "Audeo zero-shot"),
    ("foleymusic", "Foley Music (retrained)"),
    ("real_audio@shuffled", "Chance (other piece)"),
    ("silence", "Silence"),
]
COLS = [("onset_f50", "Note F1@50", 100), ("onset_f100", "Note F1@100", 100), ("frame_f", "Frame F1", 100),
        ("timing_f50", "Timing F1", 100), ("chroma_sim", "Chroma", 1), ("onset_corr", "Onset r", 1)]
TESTED = ("onset_f50", "timing_f50", "chroma_sim", "onset_corr")


def tab_column(clips_dir: Path, clip_ids: list[str], groups: list[str]) -> dict[str, float]:
    """Tab onset recall (%) per system, pooled over all tab pitches; shuffled = 30e's partner clip."""
    tabs = {c: json.loads((clips_dir / "tab" / f"{c}.json").read_text())["events"] for c in clip_ids}

    def partner(i: int) -> str:
        for k in range(3, len(clip_ids) + 3):
            j = (i + k) % len(clip_ids)
            if groups[j] != groups[i]:
                return clip_ids[j]
        raise ValueError

    out = {}
    for system, _ in ROWS:
        if system == "silence":
            out[system] = 0.0
            continue
        base = system.split("@")[0]
        hit = n = 0
        for i, c in enumerate(clip_ids):
            src = partner(i) if system.endswith("@shuffled") else c
            m = pretty_midi.PrettyMIDI(str(clips_dir / "transcriptions" / base / f"{src}.mid"))
            r = G.tab_recall(m, tabs[c], "high_first+capo")
            hit += r["onset"]
            n += r["n_tab_pitches"]
        out[system] = 100 * hit / n
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("outputs/runs"))
    p.add_argument("--tex", type=Path, default=Path("thesis/tables/baseline_results.tex"))
    args = p.parse_args()

    table, tests = [], []
    for name, runs, clips_dir, source in DATASETS:
        by = {}
        for run in runs:  # systems in several runs (real_audio, silence, ...) are identical; the first one is kept
            with open(args.runs / run / "per_clip.csv") as f:
                rows = list(csv.DictReader(f))
            new = {}
            for r in rows:
                new.setdefault(r["system"], {})[r["clip_id"]] = r
            if by and set(new["real_audio"]) != set(by["real_audio"]):
                raise ValueError(f"{run}: other clips than {runs[0]}")
            for system, clips in new.items():
                by.setdefault(system, clips)
        clip_ids = list(by["real_audio"])
        groups = [by["real_audio"][c]["group"] for c in clip_ids]
        tab = tab_column(clips_dir, clip_ids, groups) if (clips_dir / "tab").is_dir() else {}
        sig = {}  # system -> metric -> matched video significantly better than another piece's video
        for system, _ in ROWS:
            if system == "real_audio" or f"{system}@shuffled" not in by:
                continue
            sig[system] = {}
            for k in TESTED:
                a = np.array([float(by[system][c][k]) for c in clip_ids])
                b = np.array([float(by[f"{system}@shuffled"][c][k]) for c in clip_ids])
                pval = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
                sig[system][k] = pval < 0.05 and a.mean() > b.mean()
                tests.append({"dataset": name, "system": system, "metric": k, "matched": round(a.mean(), 4),
                              "mismatched": round(b.mean(), 4), "better_share": round(float(np.mean(a > b)), 3),
                              "p": f"{pval:.2g}"})
        for system, label in ROWS:
            vals = {k: float(np.mean([float(by[system][c][k]) for c in clip_ids])) * s for k, _, s in COLS}
            pseudo_ref = system == "real_audio" and source != "real MIDI"
            table.append({"dataset": name, "label_source": source, "clips": len(clip_ids), "system": label,
                          **{k: (None if pseudo_ref else vals[k]) for k, _, _ in COLS},
                          "tab_recall": tab.get(system), "notes_per_clip": float(np.mean([float(by[system][c]["n_est"]) for c in clip_ids])),
                          "sig": sig.get(system, {})})

    def cell(r, k, fmt):
        v = r[k]
        if v is None:
            return "--" if k != "onset_f50" else "(ref.)"
        s = fmt.format(v + 0.0)
        if float(s) == 0:  # no "-0.00"
            s = fmt.format(0.0)
        return s + ("*" if r["sig"].get(k) else "")

    fmts = {k: ("{:.1f}" if s == 100 else "{:.2f}") for k, _, s in COLS}
    # markdown
    head = "| Dataset | System | " + " | ".join(h for _, h, _ in COLS) + " | Tab recall | Notes/clip |"
    md = [head, "|" + "---|" * (len(COLS) + 4)]
    for r in table:
        tabv = "--" if r["tab_recall"] is None else f"{r['tab_recall']:.1f}"
        md.append(f"| {r['dataset']} ({r['label_source']}, {r['clips']} clips) | {r['system']} | "
                  + " | ".join(cell(r, k, fmts[k]) for k, _, _ in COLS) + f" | {tabv} | {r['notes_per_clip']:.1f} |")
    md_text = ("Baseline results (mean over clips; F1 in %, onset tolerance 50 / 100 ms, frame F1 at 25 fps; timing F1 "
               "ignores pitch; chroma = chroma similarity; onset r = correlation of onset curves; tab recall = share of "
               "GuiFiR's manual tab pitches found as onsets, ±100 ms). GAPS is scored against real MIDI; GuiFiR-Vid and "
               "magcil against the transcription of the real audio (pseudo-GT), whose own row is therefore omitted. "
               "* = the model is better with the clip's own video than with another piece's video, paired Wilcoxon p < 0.05 "
               "(not tested for MMAudio + prompt).\n\n" + "\n".join(md) + "\n")

    # LaTeX (booktabs)
    cols = "l" + "r" * (len(COLS) + 1)
    ncol = len(COLS) + 2
    tex = [r"% generated by scripts/30i_results_table.py -- do not edit by hand", r"\begin{table}[t]", r"\centering",
           r"\small", r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
           # two header lines with short labels, so the table fits \textwidth at \small without scaling
           # dataset = a header row above its block (no separate column), so the table fits \textwidth
           r" & \multicolumn{2}{c}{Note F1} & Frame & Timing & Chroma & Onset & Tab \\",
           r"\cmidrule(lr){2-3}",
           r"System & 50\,ms & 100\,ms & F1 & F1 & sim. & $r$ & rec. \\"]
    for i, (name, _, _, source) in enumerate(DATASETS):
        rs = [r for r in table if r["dataset"] == name]
        tex += [r"\midrule",
                rf"\multicolumn{{{ncol}}}{{l}}{{\textbf{{{name}}} \enspace\footnotesize({source}, {rs[0]['clips']} clips)}} \\"]
        for r in rs:
            vals = [cell(r, k, fmts[k]).replace("*", r"$^{*}$") for k, _, _ in COLS]
            vals = [("$-$" + v[1:]) if v.startswith("-") and v != "--" else v for v in vals]
            tabv = "--" if r["tab_recall"] is None else f"{r['tab_recall']:.1f}"
            tex.append(rf"\quad {r['system']} & " + " & ".join(vals) + f" & {tabv} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Existing models on silent guitar video. Mean over clips (8\,s, or the whole recording if "
            r"shorter: magcil bends $\approx$3\,s; 25\,fps; only frames with a "
            r"visible fretboard). F1 in \%, onset tolerance 50/100\,ms; timing F1 ignores pitch; chroma = chroma "
            r"similarity; onset $r$ = correlation of onset curves; tab rec.\ = share of GuiFiR's manual tablature "
            r"pitches found as onsets ($\pm$100\,ms). GAPS is scored against real MIDI, GuiFiR-Vid and magcil against "
            r"the transcription of the real audio (pseudo-GT; its own row is omitted there). Real audio on GAPS = "
            r"transcription ceiling. MMAudio, ThinkSound, FoleyCrafter: video $\to$ audio, scored via audio "
            r"transcription; V2N, Audeo: piano video $\to$ MIDI models on the rectified fretboard strip; "
            r"Foley Music: body and hand keypoints $\to$ MIDI, retrained by us on its own MUSIC guitar data. "
            r"Chance = real-audio transcription of another piece. $^{*}$ The model is better with the clip's own "
            r"video than with the video of another piece (paired Wilcoxon, $p<0.05$; not tested for MMAudio + "
            r"prompt).}",
            r"\label{tab:baseline_results}", r"\end{table}", ""]
    tex_text = "\n".join(tex).replace(r"(ref.)", "--")

    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(tex_text)
    out = args.runs / f"{date.today().isoformat()}_results_table"
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text(md_text)
    with open(out / "table.csv", "w", newline="") as f:
        fields = ["dataset", "label_source", "clips", "system"] + [k for k, _, _ in COLS] + ["tab_recall", "notes_per_clip"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    with open(out / "tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0]))
        w.writeheader()
        w.writerows(tests)
    print(md_text)
    print("Own video vs another piece's video:")
    for t in tests:
        print(f"  {t['dataset']:11s} {t['system']:22s} {t['metric']:11s} {t['matched']:.3f} vs "
              f"{t['mismatched']:.3f}  p={t['p']}")
    print(f"\nwritten: {args.tex}, {out}/table.md, table.csv, tests.csv")


if __name__ == "__main__":
    main()
