"""CP17: whole GAPS performances -> V2N training data (HDF5 of JPEG fretboard strips + TSV note labels).

Per piece (``process_piece``):
    1. sync: lag of the video's audio vs the GAPS wav at 3 positions (``piece_sync``); video time t = GAPS time
       t - lag, so a GAPS note at g sits at video time g + lag (same convention as the eval clips, CP2)
    2. pass 1: decode the video at 25 fps (same frame grid as ``video.detect_cuts_file``), keep the clicked
       shots as grayscale frames scaled to <= ``TRACK_W`` px width, find the clicked corners in every frame of
       the shot by matching it directly to the clicked frame (``fretboard_tracking.track_corners_keyframe``; the
       CP5 frame-to-frame tracker drifts off the neck over minutes, see 35b_check_long_tracking.py). Downscaled
       because a shot can be minutes long (max 866 s); 1280 px gave hardly more matches at 10x the time
    3. records: contiguous runs of matched frames inside a shot (``valid_runs``: gaps <= 5 frames are
       interpolated by the tracker and bridged, longer ones split the run, 2 frames dropped at each shot edge,
       runs >= 2 s). Frames that do not match the clicked view (zoomed far, dissolve, other scene) are left out
    4. pass 2: decode again at full resolution, warp every record frame to both strip versions (CP5 fretboard
       nut -> fret 20, CP12 full string nut -> bridge), JPEG q95 (V2N's setting) into one HDF5 file per piece and
       version; labels per record as V2N TSV
``build_master`` then writes one HDF5 per split whose groups are external links to the per-piece files, so V2N's
``VPTDataset`` sees a single file (``<root>/<version>/processed_v2/<split>/guitar_v2.hdf5``) and a crash only
loses the piece being written.

V2N layout (``data/preprocess_video.py``, ``data/dataset.py`` at commit a6408c4): HDF5 group per record with a
``frames`` dataset of JPEG bytes (vlen uint8) and attrs ``n_frames``, ``duration``, ``actual_fps``, ``tsv_path``;
TSV with one header row and columns onset, key_offset, ped_offset, note, velocity (seconds from record start).
Velocity is a dummy (64; GAPS velocities are not real and V2N's onset/frame model ignores it), ped_offset =
key_offset (no pedal on a guitar).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from guitar2midi.preprocessing import fretboard_tracking as T
from guitar2midi.preprocessing.fretboard_grid import extend_corners

FPS = 25.0
TRACK_W = 640
MIN_RECORD = 50     # frames (2 s)
MAX_GAP = 5         # lost-tracking frames bridged inside a record
EDGE = 2            # frames dropped at each shot boundary (cut detection is +-1 frame)
JPEG_Q = 95
VELOCITY = 64
VERSIONS = ("fret", "full")  # CP5 nut -> fret 20, CP12 nut -> bridge
H5_NAME = "guitar_v2.hdf5"


# ---------------------------------------------------------------- video
def video_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    w, h = out.strip().split("\n")[0].split(",")[:2]
    return int(w), int(h)


def iter_frames(path: Path, size: tuple[int, int] | None = None, fps: float = FPS):
    """BGR frames at ``fps`` (frame t = time t / fps, the ``detect_cuts_file`` grid), optionally scaled to
    ``size`` = (w, h) with area interpolation."""
    w, h = size or video_size(path)
    vf = f"fps={fps}" + (f",scale={w}:{h}:flags=area" if size else "")
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)  # stopping early makes ffmpeg report a broken pipe
    n = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def track_size(w: int, h: int, max_w: int = TRACK_W) -> tuple[int, int]:
    if w <= max_w:
        return w, h
    return max_w, int(round(h * max_w / w / 2)) * 2


# ---------------------------------------------------------------- records
def valid_runs(tracked: np.ndarray, max_gap: int = MAX_GAP, edge: int = EDGE,
               min_len: int = MIN_RECORD) -> list[tuple[int, int]]:
    """[start, end) runs (indices into ``tracked``, one shot) of tracked frames; untracked gaps <= ``max_gap``
    inside a run are bridged, longer gaps split it; ``edge`` frames dropped at both shot ends."""
    ok = np.asarray(tracked, bool).copy()
    ok[:edge] = False
    if edge:
        ok[len(ok) - edge:] = False
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return []
    runs, a, prev = [], idx[0], idx[0]
    for t in idx[1:]:
        if t - prev - 1 > max_gap:
            runs.append((a, prev + 1))
            a = t
        prev = t
    runs.append((a, prev + 1))
    return [(int(a), int(b)) for a, b in runs if b - a >= min_len]


# ---------------------------------------------------------------- labels
def load_notes(midi_file: Path) -> np.ndarray:
    """(N, 3) onset, offset, pitch of all non-drum notes, sorted by onset."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_file))
    notes = [(n.start, n.end, n.pitch) for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    arr = np.array(notes, dtype=np.float64).reshape(-1, 3)
    return arr[np.argsort(arr[:, 0], kind="stable")]


def record_notes(notes: np.ndarray, lag: float, t0: float, t1: float) -> np.ndarray:
    """Notes sounding in video time [t0, t1) (GAPS time + lag), times relative to t0. Notes that started before
    t0 are kept (negative onset): the frame label then starts at frame 0, as in V2N's labeler."""
    on, off = notes[:, 0] + lag, notes[:, 1] + lag
    keep = (on < t1) & (off > t0)
    return np.column_stack([on[keep] - t0, off[keep] - t0, notes[keep, 2]])


def write_gt_midi(rel_notes: np.ndarray, duration: float, dst: Path) -> None:
    """GT MIDI of a record for V2N's eval callback (``gt_dir/<record>.mid``): notes with onset in
    [0, duration), offsets clipped to the record end."""
    import pretty_midi

    dst.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24, name="guitar")
    for on, off, p in rel_notes:
        if 0 <= on < duration:
            inst.notes.append(pretty_midi.Note(VELOCITY, int(p), float(on), float(min(max(off, on + 0.01), duration))))
    pm.instruments.append(inst)
    pm.write(str(dst))


def write_tsv(rel_notes: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        f.write("onset\tkey_offset\tped_offset\tnote\tvelocity\n")
        for on, off, p in rel_notes:
            f.write(f"{on:.6f}\t{off:.6f}\t{off:.6f}\t{int(p)}\t{VELOCITY}\n")


# ---------------------------------------------------------------- sync
def piece_sync(wav: Path, video: Path, duration: float, positions=(0.2, 0.5, 0.8), length: float = 20.0,
               max_lags=(0.5, 30.0), min_corr: float = 0.5, max_spread: float = 0.02) -> dict:
    """Lag (s) of the video's audio vs the GAPS wav, measured at ``positions`` of the piece. Tries the small
    search range first (codec priming, 0-49 ms in CP2), then a wide one (videos trimmed at the start).
    ok = every window correlates >= ``min_corr`` and the lags agree within ``max_spread`` s."""
    from guitar2midi.baselines.clips import measure_sync

    res = {}
    for max_lag in max_lags:
        meas = [measure_sync(wav, video, p * duration, length, max_lag=max_lag) for p in positions]
        lags = np.array([m[0] for m in meas])
        corrs = np.array([m[1] for m in meas])
        ok = bool(np.all(np.isfinite(lags)) and corrs.min() >= min_corr and np.ptp(lags) <= max_spread)
        res = {"lag": float(np.median(lags)), "spread": float(np.ptp(lags)), "min_corr": float(corrs.min()),
               "max_lag_searched": max_lag, "ok": ok}
        if ok:
            break
    return res


# ---------------------------------------------------------------- per piece
@dataclass
class Shot:
    name: str              # <piece>__s<k>
    start: int             # [start, end) 25 fps frames in the whole video
    end: int
    ref: int               # clicked frame
    corners: np.ndarray    # clicked corners (nut + fret 20), full-resolution pixels
    tracked_corners: np.ndarray | None = None   # (end-start, 4, 2) full resolution
    tracked: np.ndarray | None = None
    runs: list = field(default_factory=list)    # [start, end) in whole-video frames


def track_shots(video: Path, shots: list[Shot]) -> tuple[int, tuple[int, int]]:
    """Pass 1: fills ``tracked_corners`` / ``tracked`` / ``runs`` of each shot. Returns (n_frames, (w, h))."""
    w, h = video_size(video)
    tw, th = track_size(w, h)
    sx, sy = tw / w, th / h
    scale = np.array([sx, sy])
    buf = {i: [] for i in range(len(shots))}
    n = 0
    for t, frame in enumerate(iter_frames(video, (tw, th))):
        for i, s in enumerate(shots):
            if s.start <= t < s.end:
                buf[i].append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        n = t + 1
    for i, s in enumerate(shots):
        frames = buf.pop(i)
        if len(frames) != s.end - s.start:  # video shorter than the shot list (should not happen)
            s.end = s.start + len(frames)
        if not s.start <= s.ref < s.end:
            continue
        c, tracked, _ = T.track_corners_keyframe(frames, s.ref - s.start, s.corners * scale)
        s.tracked_corners = c / scale
        s.tracked = tracked
        s.runs = [(s.start + a, s.start + b) for a, b in valid_runs(tracked)]
        del frames
    return n, (w, h)


def write_records(video: Path, shots: list[Shot], piece: str, out_root: Path, tsv_dir: Path,
                  notes: np.ndarray, lag: float, check_every_s: float = 30.0) -> tuple[list[dict], list]:
    """Pass 2: both strip versions of every record into ``<out_root>/<version>/pieces/<piece>.h5``, labels into
    ``tsv_dir/<record>.tsv``. Returns (record rows, [(record, check image)]: frame with quad + both strips, every
    ``check_every_s`` s of each record plus its first and last frame)."""
    import h5py

    recs = []  # (name, shot, a, b)
    for s in shots:
        for j, (a, b) in enumerate(s.runs):
            recs.append((f"{s.name.replace('__', '_')}_r{j}", s, a, b))
    rows, checks = [], []
    if not recs:
        return rows, checks
    files, dsets = {}, {}
    vlen = h5py.special_dtype(vlen=np.dtype("uint8"))
    for v in VERSIONS:
        p = out_root / v / "pieces" / f"{piece}.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        files[v] = h5py.File(p.with_suffix(".tmp"), "w")
    for name, s, a, b in recs:
        t0, t1 = a / FPS, b / FPS
        rel = record_notes(notes, lag, t0, t1)
        tsv = (tsv_dir / f"{name}.tsv").resolve()
        write_tsv(rel, tsv)
        write_gt_midi(rel, t1 - t0, tsv_dir.parent / "MIDI" / f"{name}.mid")
        for v in VERSIONS:
            g = files[v].create_group(name)
            dsets[v, name] = g.create_dataset("frames", (b - a,), dtype=vlen)
            g.attrs.update({"n_frames": b - a, "duration": (b - a) / FPS, "actual_fps": FPS,
                            "tsv_path": str(tsv), "video_start_frame": a, "shot": s.name})
        rows.append({"record": name, "piece": piece, "shot": s.name, "start_frame": a, "n_frames": b - a,
                     "seconds": round((b - a) / FPS, 2), "n_notes": int(len(rel)),
                     "onsets_inside": int(((rel[:, 0] >= 0) & (rel[:, 0] < t1 - t0)).sum()) if len(rel) else 0,
                     "lost_frames": int((~s.tracked[a - s.start:b - s.start]).sum())})
    step = int(check_every_s * FPS)
    want = {}
    for name, s, a, b in recs:
        for t in sorted({a, *range(a + step, b, step), b - 1}):
            want[t] = (name, s)
    by_frame = {}
    for name, s, a, b in recs:
        for t in range(a, b):
            by_frame[t] = (name, s, a)
    last = max(b for *_, b in recs)
    for t, frame in enumerate(iter_frames(video)):
        if t >= last:
            break
        if t not in by_frame:
            continue
        name, s, a = by_frame[t]
        c = s.tracked_corners[t - s.start]
        strips = {"fret": T.warp_strip(frame, c), "full": T.warp_strip(frame, extend_corners(c))}
        for v in VERSIONS:
            ok, enc = cv2.imencode(".jpg", strips[v], [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            dsets[v, name][t - a] = np.frombuffer(enc.tobytes(), np.uint8)
        if t in want:
            checks.append((name, check_image(frame, c, strips, f"{name} t={t / FPS:.0f}s ref={s.ref / FPS:.0f}s"
                                             + ("" if s.tracked[t - s.start] else " LOST"))))
    for v in VERSIONS:
        files[v].close()
        tmp = out_root / v / "pieces" / f"{piece}.tmp"
        tmp.rename(tmp.with_suffix(".h5"))
    return rows, checks


def check_image(frame: np.ndarray, corners: np.ndarray, strips: dict, label: str, width: int = 400) -> np.ndarray:
    """Frame with the tracked neck quad (green) and the extended nut -> bridge quad (yellow) above both strips."""
    img = frame.copy()
    for c, col in ((extend_corners(corners), (0, 255, 255)), (corners, (0, 255, 0))):
        cv2.polylines(img, [T._quad(c).astype(np.int32)], True, col, max(2, frame.shape[1] // 400))
    img = cv2.resize(img, (width, int(round(frame.shape[0] * width / frame.shape[1]))))
    st = [cv2.cvtColor(cv2.resize(strips[v], (width, 72)), cv2.COLOR_GRAY2BGR) for v in VERSIONS]
    out = np.vstack([img, *st])
    cv2.putText(out, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def build_master(piece_files: dict[str, Path], dst: Path) -> int:
    """HDF5 at ``dst`` whose groups are external links to the record groups of ``piece_files`` (record name ->
    absolute per-piece file). Returns the number of records."""
    import h5py

    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst, "w") as hf:
        hf.attrs.update({"target_fps": FPS, "target_width": T.STRIP_W, "target_height": T.STRIP_H, "grayscale": True})
        for name, path in sorted(piece_files.items()):
            hf[name] = h5py.ExternalLink(str(Path(path).resolve()), f"/{name}")
    return len(piece_files)


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
