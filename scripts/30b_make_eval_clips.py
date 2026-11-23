"""CP2: cut evaluation clips (silent video + real audio + GT MIDI) from the GAPS test split.

    python scripts/30b_make_eval_clips.py                     # 3 clips x 8 s per test piece
    python scripts/30b_make_eval_clips.py --plots 3           # also 3 frame + piano-roll figures
    python scripts/30b_make_eval_clips.py --reject 358_441wc_c1 208_c41wc_c1   # replace bad clips
    python scripts/30b_make_eval_clips.py --set 358_441wc_c1=27                # fix a window by hand

Output: data/interim/baseline_clips/
    clips.csv            one row per clip (window, notes, sync lag/correlation, keep + reason)
    video/ audio/ midi/  per-clip files (see guitar2midi.baselines.clips)
    contact_sheet.jpg    3 frames per clip (start, middle, end; red border = not kept)
    plots/<clip>.png     frame + GT piano roll
    rejected_windows.txt windows marked bad: "<piece_id> <start_s>  # <clip_id>"
    manual_windows.txt   windows set by hand: "<clip_id> <start_s>" (override the random choice)

Bad clip (e.g. no guitar visible): ``--reject <clip_id> ...`` stores its current
window in rejected_windows.txt and draws a new window from the same part of the
piece, so the clip count stays the same. Repeat if the new window is also bad. If the guitar is only visible in a few
parts of a video (slideshow-style videos), set the window by hand with
``--set <clip_id>=<start_s>`` after looking at the video.
To drop a whole piece instead, list its id in exclude.txt ('#' comments allowed).
Existing clip files are reused unless their window changed.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pretty_midi

from guitar2midi.baselines import clips as C
from guitar2midi.data import gaps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/interim/baseline_clips"))
    p.add_argument("--gaps-root", type=Path, default=gaps.GAPS_ROOT)
    p.add_argument("--clips-per-piece", type=int, default=3)
    p.add_argument("--length", type=float, default=8.0)
    p.add_argument("--min-notes", type=int, default=8)
    p.add_argument("--max-lag-ms", type=float, default=200.0,
                   help="lags are corrected; only larger |lag| is treated as a failed measurement")
    p.add_argument("--min-corr", type=float, default=0.5, help="drop clips with weaker sync correlation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plots", type=int, default=3, help="number of frame + piano-roll figures")
    p.add_argument("--frames-per-clip", type=int, default=3, help="frames per clip in the contact sheet")
    p.add_argument("--set", nargs="+", default=[], metavar="CLIP_ID=START",
                   help="set the window start (s, video time) of these clips by hand")
    p.add_argument("--reject", nargs="+", default=[], metavar="CLIP_ID",
                   help="mark the current window of these clips as bad and replace it")
    args = p.parse_args()

    # previous run: current windows (to reuse unchanged clip files and to resolve --reject)
    prev_starts = {}
    if (args.out / "clips.csv").exists():
        with open(args.out / "clips.csv") as f:
            prev_starts = {r["clip_id"]: float(r["start_s"]) for r in csv.DictReader(f)}
    rejected_file = args.out / "rejected_windows.txt"
    if args.reject:
        unknown = [c for c in args.reject if c not in prev_starts]
        if unknown:
            raise SystemExit(f"--reject: unknown clip ids {unknown} (not in {args.out / 'clips.csv'})")
        rejected_file.parent.mkdir(parents=True, exist_ok=True)
        with open(rejected_file, "a") as f:
            for cid in args.reject:
                f.write(f"{cid.rsplit('_c', 1)[0]} {prev_starts[cid]}  # {cid}\n")
    manual_file = args.out / "manual_windows.txt"
    if args.set:
        with open(manual_file, "a") as f:
            for item in args.set:
                cid, start = item.split("=")
                f.write(f"{cid} {float(start)}\n")
    manual: dict[str, float] = {}  # later lines win
    if manual_file.exists():
        for ln in manual_file.read_text().splitlines():
            parts = ln.split("#")[0].split()
            if len(parts) == 2:
                manual[parts[0]] = float(parts[1])
    rejected: dict[str, list[float]] = {}
    if rejected_file.exists():
        for ln in rejected_file.read_text().splitlines():
            parts = ln.split("#")[0].split()
            if len(parts) == 2:
                rejected.setdefault(parts[0], []).append(float(parts[1]))

    exclude_file = args.out / "exclude.txt"
    excluded = set()
    if exclude_file.exists():
        excluded = {ln.split("#")[0].strip() for ln in exclude_file.read_text().splitlines()} - {""}

    rows = []
    pieces = gaps.load_metadata("test", args.gaps_root)
    for pi, piece in enumerate(pieces):
        video = gaps.video_path(piece, args.gaps_root)
        wav = gaps.audio_path(piece, args.gaps_root)
        if not video.exists():
            print(f"{piece['id']}: no video, skipped")
            continue
        midi = pretty_midi.PrettyMIDI(str(gaps.midi_path(piece, args.gaps_root)))
        info = gaps.video_info(video)
        duration = min(gaps.wav_duration(wav), info["duration"])
        rng = np.random.default_rng(args.seed + pi)
        starts = C.select_windows(C.note_onsets(midi), duration, args.clips_per_piece, args.length,
                                  args.min_notes, rng=rng, rejected=rejected.get(piece["id"], []))
        starts = [manual.get(f"{piece['id']}_c{k}", s0) for k, s0 in enumerate(starts, 1)]
        for k, start in enumerate(starts, 1):
            cid = f"{piece['id']}_c{k}"
            v_out, a_out, m_out = (args.out / d / f"{cid}.{e}" for d, e in
                                   (("video", "mp4"), ("audio", "wav"), ("midi", "mid")))
            # Lag of the video's own audio vs the GAPS wav. These are constant per piece and match
            # audio-codec priming delays (576/1024 samples...), so we correct instead of dropping:
            # video time t corresponds to GAPS time t - lag (audio and MIDI are cut there).
            lag, corr = C.measure_sync(wav, video, start, args.length)
            gaps_start = start - (lag if np.isfinite(lag) else 0.0)
            if not v_out.exists() or prev_starts.get(cid) != round(start, 3):
                C.cut_video(video, v_out, start, args.length)
            C.cut_wav(wav, a_out, gaps_start, args.length)
            n_notes = C.cut_midi(midi, m_out, gaps_start, args.length)

            reasons = []
            if not abs(lag) * 1000 <= args.max_lag_ms:
                reasons.append(f"sync lag {lag * 1000:.0f} ms (unreliable)")
            if corr < args.min_corr:
                reasons.append(f"sync corr {corr:.2f}")
            if n_notes < args.min_notes:
                reasons.append(f"only {n_notes} notes")
            if piece["id"] in excluded or cid in excluded:
                reasons.append("excluded manually")
            rows.append({
                "clip_id": cid, "piece_id": piece["id"], "performer": piece["performer_name"],
                "start_s": round(start, 3), "end_s": round(start + args.length, 3),
                "gaps_start_s": round(gaps_start, 4), "n_notes": n_notes,
                "sync_lag_ms": round(lag * 1000, 1), "sync_corr": round(corr, 3),
                "src_fps": round(info["fps"], 3), "resolution": f"{info['width']}x{info['height']}",
                "keep": int(not reasons), "reason": "; ".join(reasons),
                "window": "manual" if cid in manual else "random",
            })
            print(f"{cid}: {start:7.1f}s  notes={n_notes:3d}  lag={lag * 1000:6.1f} ms  corr={corr:.2f}  "
                  f"{'keep' if not reasons else 'DROP: ' + rows[-1]['reason']}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    C.contact_sheet([(args.out / "video" / f"{r['clip_id']}.mp4",
                      f"{r['clip_id']} n={r['n_notes']} lag={r['sync_lag_ms']:.0f}ms", bool(r["keep"]))
                     for r in rows], args.out / "contact_sheet.jpg", frames_per_clip=args.frames_per_clip)
    kept = [r for r in rows if r["keep"]]
    for old in (args.out / "plots").glob("*.png"):
        old.unlink()
    for r in kept[:: max(len(kept) // max(args.plots, 1), 1)][: args.plots]:
        C.plot_clip(args.out / "video" / f"{r['clip_id']}.mp4", args.out / "midi" / f"{r['clip_id']}.mid",
                    args.out / "plots" / f"{r['clip_id']}.png", f"{r['clip_id']} ({r['performer']})")

    lags = np.array([r["sync_lag_ms"] for r in rows])
    print("\n== CP2 summary ==")
    print(f"clips:        {len(rows)} from {len({r['piece_id'] for r in rows})} pieces")
    print(f"kept:         {len(kept)} from {len({r['piece_id'] for r in kept})} pieces "
          f"({len(kept) * args.length / 60:.1f} min)")
    print(f"sync lag ms:  median {np.nanmedian(lags):.1f}, max |lag| {np.nanmax(np.abs(lags)):.1f}")
    print(f"notes/clip:   median {int(np.median([r['n_notes'] for r in rows]))}")
    changed = [r["clip_id"] for r in rows if prev_starts and prev_starts.get(r["clip_id"]) != r["start_s"]]
    if changed:
        print(f"new windows:  {', '.join(changed)}")
    if rejected:
        print(f"rejected:     {sum(map(len, rejected.values()))} windows (see {rejected_file})")
    for r in rows:
        if not r["keep"]:
            print(f"  dropped {r['clip_id']}: {r['reason']}")
    print(f"outputs:      {args.out}/clips.csv, contact_sheet.jpg, plots/")


if __name__ == "__main__":
    main()
