"""CP8: magcil evaluation clips (silent video + real audio), 3 per technique.

    python scripts/31b_make_magcil_clips.py                 # env guitar2midi

Selection per technique: one recording for each guitar/amp pair (carvin/dc-modern, palm-bay/plexiglass,
rwg/sl100), so the 3 clips cover all guitars and amps; inside a pair the longest recording (ties: first name).
Window: the centred ``--length`` s (default 8 s); recordings shorter than that are used whole (bend recordings
are 2-5 s), so clips.csv has a ``length_s`` column. Audio = the dataset wav (extracted from the same mp4), cut at
the same window after a sync check against the mp4's own audio.

Output: data/interim/baseline_clips_magcil/
    clips.csv  video/ audio/  contact_sheet.jpg

Next:
    python scripts/30d_transcribe.py --clips-dir data/interim/baseline_clips_magcil --system real_audio  # env amt
    python scripts/31b_spot_check_magcil.py
    python scripts/30f_prepare_strip_frames.py --clips-dir data/interim/baseline_clips_magcil
    (click corners, then) python scripts/30f_make_v2n_strips.py --clips-dir data/interim/baseline_clips_magcil \\
        --annotations data/labels/fretboard_corners/magcil_baseline_EVAL_ONLY.jsonl
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from guitar2midi.baselines import clips as C
from guitar2midi.baselines.clips_guifir import video_props
from guitar2midi.data import magcil

PAIRS = (("carvin", "dc-modern"), ("palm-bay", "plexiglass"), ("rwg", "sl100"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/interim/baseline_clips_magcil"))
    p.add_argument("--length", type=float, default=8.0)
    p.add_argument("--edge", type=float, default=0.1, help="s cut from both ends of short recordings")
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--min-corr", type=float, default=0.5)
    args = p.parse_args()

    recs = magcil.list_recordings()
    rows = []
    for k, tech in magcil.TECHNIQUES.items():
        for guitar, amp in PAIRS:
            cands = [r for r in recs if r["class"] == k and r["guitar"] == guitar and r["amp"] == amp]
            for r in cands:
                r["props"] = video_props(r["video"])
            r = max(cands, key=lambda r: (r["props"]["duration"], -cands.index(r)))
            dur = r["props"]["duration"]
            length = min(args.length, dur - 2 * args.edge)
            start = (dur - length) / 2
            cid = f"{r['stem']}_m1"
            v_out, a_out = args.out / "video" / f"{cid}.mp4", args.out / "audio" / f"{cid}.wav"
            lag, corr = C.measure_sync(r["wav"], r["video"], start, length, max_lag=min(0.5, start))
            wav_start = start - (lag if np.isfinite(lag) else 0.0)
            C.cut_video(r["video"], v_out, start, length, max_width=args.max_width)
            C.cut_wav(r["wav"], a_out, wav_start, length)
            reasons = [] if corr >= args.min_corr else [f"sync corr {corr:.2f}"]
            rows.append({"clip_id": cid, "technique": tech, "class": k, "guitar": guitar, "amp": amp,
                         "exercise": r["exercise"], "rec_duration_s": round(dur, 3), "start_s": round(start, 3),
                         "length_s": round(length, 3), "sync_lag_ms": round(lag * 1000, 1),
                         "sync_corr": round(corr, 3), "src_resolution": f"{r['props']['width']}x{r['props']['height']}",
                         "keep": int(not reasons), "reason": "; ".join(reasons)})
            print(f"{cid:42s} {tech:17s} {length:4.1f}s of {dur:5.1f}s  lag {lag * 1000:6.1f} ms  corr {corr:.2f}"
                  f"{'  DROP: ' + rows[-1]['reason'] if reasons else ''}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    C.contact_sheet([(args.out / "video" / f"{r['clip_id']}.mp4", f"{r['technique']} {r['guitar']} {r['length_s']:.1f}s",
                      bool(r["keep"])) for r in rows], args.out / "contact_sheet.jpg")
    lens = np.array([r["length_s"] for r in rows])
    print("\n== CP8 clips summary ==")
    print(f"clips:      {len(rows)} ({sum(r['keep'] for r in rows)} kept), {lens.sum() / 60:.1f} min, "
          f"length median {np.median(lens):.1f} s, {int((lens < args.length).sum())} shorter than {args.length:.0f} s")
    print(f"sync lag:   max |lag| {np.nanmax(np.abs([r['sync_lag_ms'] for r in rows])):.1f} ms, "
          f"min corr {min(r['sync_corr'] for r in rows):.2f}")
    print(f"outputs:    {args.out}/clips.csv, contact_sheet.jpg")


if __name__ == "__main__":
    main()
