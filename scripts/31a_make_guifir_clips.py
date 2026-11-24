"""CP7: GuiFiR-Vid evaluation clips (silent video + real audio + tab) and V2N strips from the GT corners.

    python scripts/31a_make_guifir_clips.py                 # env guitar2midi

One clip per song from the GuiFiR-Vid **test** split (the clip with the most tab events in its best 8 s window).
No clicking: the fretboard tracker (``fretboard_tracking.track_corners``, same as CP5) is seeded with the GuiFiR
GT corners of the tab frame closest to the window centre. The other GT-corner frames in the window measure the
tracking error. GuiFiR test labels are evaluation-only (never FPE training data).

Output: data/interim/baseline_clips_guifir/
    clips.csv                 one row per clip (source clip, window, tab events, resolution, cuts, tracking error)
    video/ audio/ tab/        per-clip files (see guitar2midi.baselines.clips_guifir)
    contact_sheet.jpg         3 frames per clip
    strips/<clip>.npy         (T, 144, 800) uint8 strips at 25 fps, black outside the seeded shot
    strips/<clip>_valid.npy   (T,) bool, frames inside the seeded shot
    strips/tracking_sheet.jpg tracked neck + strips, 3 frames per clip

Next: reference transcription of the real audio (env amt)
    python scripts/30d_transcribe.py --clips-dir data/interim/baseline_clips_guifir --system real_audio
then the tab check: python scripts/31a_check_guifir_tab.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from guitar2midi.baselines import clips as C
from guitar2midi.baselines import clips_guifir as G
from guitar2midi.data import guifir
from guitar2midi.preprocessing import fretboard_tracking as T
from guitar2midi.preprocessing.fretboard_grid import corners_valid
from guitar2midi.preprocessing.video import detect_cuts, read_frames, shots


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/interim/baseline_clips_guifir"))
    p.add_argument("--split", default="test")
    p.add_argument("--length", type=float, default=8.0)
    p.add_argument("--max-width", type=int, default=1280, help="downscale wider videos (1080p/4K) to this width")
    p.add_argument("--min-coverage", type=float, default=0.9, help="flag clips with fewer valid strip frames")
    args = p.parse_args()

    by_clip = G.records_by_clip(args.split)
    vdir = guifir.GUIFIR_ROOT / args.split / "videos"

    # ---- pick one clip per song: most tab events inside its best window
    cand = {}
    for cid, recs in by_clip.items():
        props = G.video_props(vdir / f"{cid}.mp4")
        times = np.array([guifir.frame_index(r) / props["fps"] for r in recs])
        start = G.choose_window(times, props["duration"], args.length)
        n = int(((times >= start) & (times < start + args.length)).sum())
        cand[cid] = (n, start, props)
    chosen = {}
    for cid in sorted(cand):
        song = G.song_of(cid)
        if song not in chosen or cand[cid][0] > cand[chosen[song]][0]:
            chosen[song] = cid
    print(f"{len(by_clip)} test clips, {len(chosen)} songs -> {len(chosen)} eval clips")

    out = args.out
    (out / "strips").mkdir(parents=True, exist_ok=True)
    (out / "tab").mkdir(parents=True, exist_ok=True)
    rows, blocks = [], []
    for song in sorted(chosen):
        src_id = chosen[song]
        n_ev, start, props = cand[src_id]
        cid = f"{src_id}_g1"  # one window per source clip
        src = vdir / f"{src_id}.mp4"
        v_out, a_out = out / "video" / f"{cid}.mp4", out / "audio" / f"{cid}.wav"
        C.cut_video(src, v_out, start, args.length, max_width=args.max_width)
        G.cut_audio(src, a_out, start, args.length)
        frames = read_frames(v_out)
        h, w = frames[0].shape[:2]
        n_frames = len(frames)

        # tab events in clip time, with GT corners in clip pixels
        events = []
        for r in by_clip[src_id]:
            t = guifir.frame_index(r) / props["fps"] - start
            if not 0 <= t < args.length:
                continue
            gc = guifir.gt_corners(r)
            gc = None if gc is None else gc * (w / r["frame_width"])
            events.append({"t": round(t, 4), "frame": min(int(round(t * C.FPS)), n_frames - 1),
                           "frame_tab": r["frame_tab"], "capo": r["frame_capo"],
                           "corners": None if gc is None else gc.round(2).tolist(), "source_png": r["filename"]})
        (out / "tab" / f"{cid}.json").write_text(json.dumps({"clip_id": cid, "source_clip": src_id, "start_s": start,
                                                              "events": events}, indent=1))

        # seed = valid GT corners closest to the window centre
        seeds = [e for e in events if e["corners"] is not None and corners_valid(np.array(e["corners"]))]
        cuts = detect_cuts(frames)
        row = {"clip_id": cid, "source_clip": src_id, "song": song, "start_s": start, "end_s": start + args.length,
               "n_tab_events": len(events), "n_tab_pitches": sum(len(G.tab_pitches(e["frame_tab"], e["capo"]))
                                                                for e in events),
               "src_fps": round(props["fps"], 3), "src_resolution": f"{props['width']}x{props['height']}",
               "clip_resolution": f"{w}x{h}", "cuts": " ".join(map(str, cuts))}
        if not seeds:
            row.update(keep=0, reason="no valid GT corners in window")
            rows.append(row)
            print(f"{cid}: no valid GT corners, no strips")
            continue
        seed = min(seeds, key=lambda e: abs(e["t"] - args.length / 2))
        ref = seed["frame"]
        shot = next((a, b) for a, b in shots(n_frames, cuts) if a <= ref < b)
        corners, tracked, inliers = T.track_corners(frames, ref, np.array(seed["corners"]), shot=shot)
        valid = ~np.isnan(corners[:, 0, 0])
        strips = np.stack([T.warp_strip(f, c) for f, c in zip(frames, corners)])
        np.save(out / "strips" / f"{cid}.npy", strips)
        np.save(out / "strips" / f"{cid}_valid.npy", valid)

        # tracking error at the other GT frames (mean corner distance, px at 640x360 like GuiFiR's FCD)
        errs = [float(np.linalg.norm(corners[e["frame"]] - np.array(e["corners"]), axis=1).mean() * 640 / w)
                for e in events if e["corners"] is not None and e is not seed and valid[e["frame"]]]
        coverage = float((tracked & valid).mean())
        row.update(seed_frame=ref, shot=f"{shot[0]}-{shot[1]}", coverage=round(coverage, 3),
                   valid_frames=round(float(valid.mean()), 3),
                   track_err_median_px640=round(float(np.median(errs)), 1) if errs else "",
                   track_err_max_px640=round(float(np.max(errs)), 1) if errs else "", n_err_frames=len(errs),
                   keep=1, reason="")
        flag = coverage < args.min_coverage
        rows.append(row)
        idx = [int(round(x)) for x in np.linspace(0.05, 0.95, 3) * (n_frames - 1)]
        label = (f"{cid} cov={coverage:.0%} err={row['track_err_median_px640']}px" + ("  FLAG" if flag else ""))
        blocks.append(T.tracking_block(frames, corners, tracked, strips, idx, label, flag))
        print(f"{cid}: {start:4.2f}s  tab events {len(events):2d}  {row['src_resolution']:>9}  cuts {cuts or '-'}  "
              f"coverage {coverage:.0%}  track err median {row['track_err_median_px640']} px"
              f"{'  FLAG' if flag else ''}", flush=True)

    fields = max((list(r) for r in rows), key=len)
    with open(out / "clips.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, restval="")
        wr.writeheader()
        wr.writerows(rows)
    C.contact_sheet([(out / "video" / f"{r['clip_id']}.mp4", f"{r['clip_id']} tab={r['n_tab_events']}",
                      bool(r["keep"])) for r in rows], out / "contact_sheet.jpg")
    T.save_sheet(blocks, out / "strips" / "tracking_sheet.jpg")

    kept = [r for r in rows if r["keep"]]
    errs = [r["track_err_median_px640"] for r in kept if r["track_err_median_px640"] != ""]
    print("\n== CP7 clips summary ==")
    print(f"clips:          {len(rows)} ({len(kept)} with strips), {len(rows) * args.length / 60:.1f} min")
    print(f"tab events:     median {int(np.median([r['n_tab_events'] for r in rows]))} per clip, "
          f"{sum(r['n_tab_pitches'] for r in rows)} tab pitches in total")
    print(f"with cuts:      {sum(bool(r['cuts']) for r in rows)}")
    print(f"strip coverage: median {np.median([r['coverage'] for r in kept]):.0%}, "
          f"{sum(r['coverage'] < args.min_coverage for r in kept)} below {args.min_coverage:.0%}")
    if errs:
        print(f"tracking error: median of clip medians {np.median(errs):.1f} px (640x360), worst clip {max(errs):.1f} px")
    print(f"outputs:        {out}/clips.csv, contact_sheet.jpg, strips/tracking_sheet.jpg")


if __name__ == "__main__":
    main()
