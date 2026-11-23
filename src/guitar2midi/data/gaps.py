"""Readers and downloaders for GAPS v1.1 (Riley et al., ISMIR 2024).

Source: Hugging Face dataset ``xavriley/GAPS`` (v1.1, MIT; audio included, the
case-clashing file names of the v1 Zenodo zip are fixed). Performance videos are
not distributed; they are fetched from YouTube via ``yt_id``.

Layout after scripts/30a_download_gaps.py:
    data/raw/gaps/gaps_metadata_with_splits.csv
    data/raw/gaps/{audio,midi}/<id>.{wav,mid}      id e.g. "001_mvswc"
    data/raw/gaps/videos/<id>.mp4                   with its original audio track
    data/raw/gaps/videos/failed.tsv                 id, yt_id, error for missing videos

Splits: 270 train, 30 test. The 101 rows without a split are not part of the
GAPS benchmark and are ignored.
"""

from __future__ import annotations

import csv
import json
import subprocess
import wave
from pathlib import Path

GAPS_ROOT = Path("data/raw/gaps")
HF_REPO = "xavriley/GAPS"
METADATA_FILE = "gaps_metadata_with_splits.csv"
SPLITS = ("train", "test")


def download_metadata(root: Path = GAPS_ROOT) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, METADATA_FILE, repo_type="dataset", local_dir=root))


def load_metadata(split: str | None = None, root: Path = GAPS_ROOT) -> list[dict]:
    """Rows of the metadata CSV that belong to a GAPS split (``split=None``: train + test)."""
    with open(root / METADATA_FILE, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in SPLITS]
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
        rows = [r for r in rows if r["split"] == split]
    return rows


def audio_path(row: dict, root: Path = GAPS_ROOT) -> Path:
    return root / row["audio_path"]


def midi_path(row: dict, root: Path = GAPS_ROOT) -> Path:
    return root / row["midi_path"]


def video_path(row: dict, root: Path = GAPS_ROOT) -> Path:
    return root / "videos" / f"{row['id']}.mp4"


def download_files(rows: list[dict], kinds: tuple[str, ...] = ("audio", "midi"), root: Path = GAPS_ROOT) -> None:
    """Fetch the audio and/or MIDI of the given rows from Hugging Face (skips existing files)."""
    from huggingface_hub import hf_hub_download

    for row in rows:
        for kind in kinds:
            rel = row[f"{kind}_path"]
            if not (root / rel).exists():
                hf_hub_download(HF_REPO, rel, repo_type="dataset", local_dir=root)


def download_video(row: dict, root: Path = GAPS_ROOT, max_height: int = 720) -> str | None:
    """Download the performance video (with audio, needed for the sync check). Returns an error or None."""
    import yt_dlp

    out = video_path(row, root)
    if out.exists():
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]/b[height<={max_height}]",
        "merge_output_format": "mp4",
        "outtmpl": str(out.with_suffix(".%(ext)s")),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={row['yt_id']}"])
    except Exception as e:  # yt_dlp raises DownloadError for unavailable/private videos
        return str(e).splitlines()[0]
    return None if out.exists() else "download finished but no .mp4 was written"


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def video_info(path: Path) -> dict:
    """Duration (s), fps, width, height of a video via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(path)]
    info = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    stream = info["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    return {
        "duration": float(info["format"]["duration"]),
        "fps": float(num) / float(den) if float(den) else 0.0,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }
