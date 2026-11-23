"""CP1: download GAPS v1.1 (metadata, audio, MIDI from Hugging Face; videos from YouTube).

    python scripts/30a_download_gaps.py --split test          # baseline study: test split only (~1.2 GB audio)
    python scripts/30a_download_gaps.py --split train --no-audio   # CP8: train MIDI + videos only

Writes data/raw/gaps/ (see guitar2midi.data.gaps) and a per-piece summary
data/raw/gaps/summary_<split>.csv; prints totals. Safe to rerun (skips existing files).
"""

import argparse
import csv
from pathlib import Path

from guitar2midi.data import gaps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=gaps.SPLITS, default="test")
    p.add_argument("--root", type=Path, default=gaps.GAPS_ROOT)
    p.add_argument("--no-audio", action="store_true", help="skip GAPS audio (MIDI is always fetched)")
    p.add_argument("--no-videos", action="store_true")
    p.add_argument("--max-height", type=int, default=720)
    args = p.parse_args()

    gaps.download_metadata(args.root)
    rows = gaps.load_metadata(args.split, args.root)
    print(f"GAPS {args.split}: {len(rows)} pieces")

    kinds = ("midi",) if args.no_audio else ("audio", "midi")
    print(f"downloading {' + '.join(kinds)} from Hugging Face ...")
    gaps.download_files(rows, kinds, args.root)

    failed = {}
    if not args.no_videos:
        for i, row in enumerate(rows, 1):
            err = gaps.download_video(row, args.root, args.max_height)
            status = "ok" if err is None else f"FAILED: {err}"
            print(f"  video {i:3d}/{len(rows)} {row['id']} ({row['yt_id']}): {status}", flush=True)
            if err is not None:
                failed[row["id"]] = (row["yt_id"], err)
        failed_file = args.root / "videos" / f"failed_{args.split}.tsv"
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_file, "w") as f:
            f.writelines(f"{k}\t{y}\t{e}\n" for k, (y, e) in failed.items())

    summary = []
    for row in rows:
        rec = {"id": row["id"], "yt_id": row["yt_id"], "performer": row["performer_name"], "title": row["title"]}
        a = gaps.audio_path(row, args.root)
        rec["audio_s"] = round(gaps.wav_duration(a), 2) if a.exists() else ""
        v = gaps.video_path(row, args.root)
        if v.exists():
            info = gaps.video_info(v)
            rec.update(video_s=round(info["duration"], 2), fps=round(info["fps"], 3),
                       resolution=f"{info['width']}x{info['height']}")
        else:
            rec.update(video_s="", fps="", resolution="")
        summary.append(rec)
    out = args.root / f"summary_{args.split}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    n_vid = sum(r["video_s"] != "" for r in summary)
    audio_h = sum(r["audio_s"] or 0 for r in summary) / 3600
    video_h = sum(r["video_s"] or 0 for r in summary) / 3600
    usable_h = sum(r["audio_s"] or 0 for r in summary if r["video_s"] != "") / 3600
    print(f"\n== GAPS {args.split} summary ==")
    print(f"pieces:            {len(rows)}  (performers: {len({r['performer'] for r in summary})})")
    print(f"MIDI:              {sum(gaps.midi_path(r, args.root).exists() for r in rows)}/{len(rows)}")
    print(f"audio:             {sum(r['audio_s'] != '' for r in summary)}/{len(rows)}  ({audio_h:.2f} h)")
    if not args.no_videos:
        print(f"videos downloaded: {n_vid}/{len(rows)}  ({video_h:.2f} h of video)")
        print(f"videos missing:    {len(failed)}  (see {failed_file})")
        print(f"usable (audio+video): {usable_h:.2f} h")
    print(f"per-piece table:   {out}")


if __name__ == "__main__":
    main()
