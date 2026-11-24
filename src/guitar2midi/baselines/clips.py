"""Evaluation clips for the baseline study (CP2).

Per clip (fixed length, default 8 s = MMAudio's native length):
    video/<clip>.mp4   silent, resampled to 25 fps, original resolution (H.264)
    audio/<clip>.wav   real GAPS audio, original sample rate/channels
    midi/<clip>.mid    GT notes whose onset lies in the window, shifted to t=0,
                       offsets clipped at the window end

Sync check: the video's own audio track is cross-correlated with the GAPS wav.
A positive lag means the video is *later* than the GAPS audio (and its MIDI).
On the GAPS test split the lag is constant per piece (0-49 ms, correlation ~1)
and equals audio-codec priming delays, so clips are corrected: video window
[t, t+L) is paired with GAPS audio/MIDI [t - lag, t - lag + L).
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
import pretty_midi

FPS = 25
SYNC_SR = 16000


# ---------------------------------------------------------------- window selection
def note_onsets(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    return np.sort(np.array([n.start for inst in midi.instruments for n in inst.notes]))


def select_windows(onsets: np.ndarray, duration: float, n_clips: int = 3, length: float = 8.0,
                   min_notes: int = 8, margin: float = 1.0, step: float = 0.5,
                   rng: np.random.Generator | None = None, rejected: list[float] = ()) -> list[float]:
    """Pick ``n_clips`` non-overlapping window starts, one per equal part of the piece.

    Inside each part a start is drawn at random among windows with at least
    ``min_notes`` onsets (if none qualifies: the densest window of that part).
    ``margin`` keeps windows away from the file edges (room for the sync check).
    ``rejected``: starts of windows marked bad (e.g. no guitar visible); candidates
    overlapping them are skipped, so a rejected clip is replaced by a new window
    from the same part of the piece.
    """
    rng = rng or np.random.default_rng(0)
    lo, hi = margin, duration - margin - length
    if hi <= lo:
        return []
    starts = np.arange(lo, hi, step)
    counts = np.searchsorted(onsets, starts + length) - np.searchsorted(onsets, starts)
    chosen = []
    for part in np.array_split(np.arange(len(starts)), n_clips):
        # drop candidates overlapping the previous pick or a rejected window
        part = [i for i in part if (not chosen or starts[i] >= chosen[-1] + length)
                and all(abs(starts[i] - r) >= length for r in rejected)]
        if not part:
            continue
        ok = [i for i in part if counts[i] >= min_notes]
        i = int(rng.choice(ok)) if ok else max(part, key=lambda j: counts[j])
        chosen.append(float(starts[i]))
    return chosen


# ---------------------------------------------------------------- cutting
def cut_video(src: Path, dst: Path, start: float, length: float, fps: int = FPS,
              max_width: int | None = None) -> None:
    """Silent, frame-accurate cut resampled to ``fps`` (re-encoded H.264).
    ``max_width``: downscale wider videos to this width (aspect kept), e.g. 4K GuiFiR clips."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps}" + (f",scale='min({max_width},iw)':-2" if max_width else "")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
           "-an", "-vf", vf, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)


def cut_wav(src: Path, dst: Path, start: float, length: float) -> None:
    """Sample-exact copy of [start, start+length) keeping the original format."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(src)) as r:
        sr = r.getframerate()
        r.setpos(int(round(start * sr)))
        frames = r.readframes(int(round(length * sr)))
        params = r.getparams()
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)


def cut_midi(midi: pretty_midi.PrettyMIDI, dst: Path, start: float, length: float) -> int:
    """Notes with onset in [start, start+length), shifted to 0 and clipped at the end. Returns note count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = pretty_midi.PrettyMIDI()
    n = 0
    for inst in midi.instruments:
        new = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for note in inst.notes:
            if start <= note.start < start + length:
                new.notes.append(pretty_midi.Note(note.velocity, note.pitch, note.start - start,
                                                  min(note.end, start + length) - start))
        n += len(new.notes)
        out.instruments.append(new)
    out.write(str(dst))
    return n


# ---------------------------------------------------------------- sync check
def load_audio_mono(src: Path, start: float, length: float, sr: int = SYNC_SR) -> np.ndarray:
    """Decode [start, start+length) of any audio/video file to mono float32 via ffmpeg."""
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{max(start, 0):.3f}", "-i", str(src), "-t", f"{length:.3f}",
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)


def measure_sync(gaps_wav: Path, video: Path, start: float, length: float,
                 max_lag: float = 0.5, sr: int = SYNC_SR) -> tuple[float, float]:
    """Lag (s) of the video's audio relative to the GAPS audio, and the normalised peak correlation."""
    from scipy.signal import correlate

    ref = load_audio_mono(gaps_wav, start, length, sr)
    vid = load_audio_mono(video, start - max_lag, length + 2 * max_lag, sr)
    if len(ref) == 0 or len(vid) <= len(ref):
        return float("nan"), 0.0
    ref = ref - ref.mean()
    vid = vid - vid.mean()
    corr = correlate(vid, ref, mode="valid", method="fft")  # corr[k]: ref aligned at vid[k:]
    # normalise by the energy of each vid window
    energy = np.sqrt(np.convolve(vid ** 2, np.ones(len(ref)), mode="valid")) * np.linalg.norm(ref)
    ncc = corr / np.maximum(energy, 1e-9)
    k = int(np.argmax(ncc))
    return k / sr - max_lag, float(ncc[k])


# ---------------------------------------------------------------- visual checks
def middle_frame(video: Path) -> np.ndarray | None:
    import cv2

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 2, 0))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def frames_at(video: Path, fractions: list[float]) -> list[np.ndarray | None]:
    """Frames at the given fractions (0..1) of the clip length."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in fractions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(int(f * n), 0), max(n - 1, 0)))
        ok, frame = cap.read()
        out.append(frame if ok else None)
    cap.release()
    return out


def contact_sheet(items: list[tuple[Path, str, bool]], dst: Path, frames_per_clip: int = 3,
                  clips_per_row: int = 3, tile_w: int = 200) -> None:
    """Grid with ``frames_per_clip`` evenly spread frames per clip (start ... end), so cuts and
    non-guitar shots anywhere in the clip are visible. Label = clip id + info; red border if not kept.
    """
    import cv2

    tile_h = tile_w * 9 // 16
    gap, label_h = 8, 22
    block_w = frames_per_clip * tile_w + gap
    fractions = list(np.linspace(0.05, 0.95, frames_per_clip)) if frames_per_clip > 1 else [0.5]
    rows = (len(items) + clips_per_row - 1) // clips_per_row
    sheet = np.full((rows * (tile_h + label_h), clips_per_row * block_w, 3), 255, np.uint8)
    for i, (video, label, keep) in enumerate(items):
        r, c = divmod(i, clips_per_row)
        y, x = r * (tile_h + label_h), c * block_w
        for j, frame in enumerate(frames_at(video, fractions)):
            tile = np.zeros((tile_h, tile_w, 3), np.uint8)
            if frame is not None:  # letterbox
                s = min(tile_w / frame.shape[1], tile_h / frame.shape[0])
                f = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
                y0, x0 = (tile_h - f.shape[0]) // 2, (tile_w - f.shape[1]) // 2
                tile[y0:y0 + f.shape[0], x0:x0 + f.shape[1]] = f
            sheet[y:y + tile_h, x + j * tile_w:x + (j + 1) * tile_w] = tile
        if not keep:
            cv2.rectangle(sheet, (x, y), (x + frames_per_clip * tile_w - 1, y + tile_h - 1), (0, 0, 255), 5)
        cv2.putText(sheet, label, (x + 4, y + tile_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def plot_clip(video: Path, midi_file: Path, dst: Path, title: str = "") -> None:
    """Middle video frame next to the GT piano roll of the clip."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    midi = pretty_midi.PrettyMIDI(str(midi_file))
    notes = [n for inst in midi.instruments for n in inst.notes]
    frame = middle_frame(video)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 1.6]})
    if frame is not None:
        a0.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    a0.set_axis_off()
    a0.set_title("middle frame")
    for n in notes:
        a1.barh(n.pitch, n.end - n.start, left=n.start, height=0.8, color="tab:blue")
        a1.plot([n.start], [n.pitch], "k|", ms=8)
    a1.set_xlabel("time in clip (s)")
    a1.set_ylabel("MIDI pitch")
    a1.set_title(f"GT notes ({len(notes)}), | = onset")
    a1.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=110)
    plt.close(fig)
