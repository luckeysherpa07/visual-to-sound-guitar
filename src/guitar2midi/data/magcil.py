"""Reader for the magcil guitar_style_dataset (Mitsou et al., Data in Brief 52, 2024, doi 10.1016/j.dib.2023.109842).

549 smartphone recordings (mp4, 1080p, 30 fps) of one guitarist playing 9 electric-guitar techniques with
3 guitars (carvin, palm-bay, rwg) x 3 amp simulations (dc-modern, plexiglass, sl100). The wav files (48 kHz
stereo) were extracted from the mp4s. Labels: technique class only; MuseScore files of the exercises are
not time-aligned. Licence CC BY-NC-ND 4.0. Some mp4s carry rotate=180 metadata (applied by ffmpeg when
cutting, not by OpenCV on the raw file); some are mirrored (front camera).

Source: local zip guitar_style_dataset-v1.0.0.zip (GitHub magcil/guitar_style_dataset, commit eb27d7b),
extracted unchanged to data/raw/magcil/. File names: class_<k>_<guitar>_<amp>_<exercise...>.{mp4,wav}.
"""

from __future__ import annotations

from pathlib import Path

MAGCIL_ROOT = Path("data/raw/magcil/magcil-guitar_style_dataset-eb27d7b/data")
TECHNIQUES = {0: "alternate picking", 1: "legato", 2: "tapping", 3: "sweep picking", 4: "vibrato",
              5: "hammer on", 6: "pull off", 7: "slide", 8: "bend"}


def parse_name(stem: str) -> dict:
    """``class_0_carvin_dc-modern_1_2`` -> class, guitar, amp, exercise."""
    parts = stem.split("_")
    k = int(parts[1])
    return {"class": k, "technique": TECHNIQUES[k], "guitar": parts[2], "amp": parts[3],
            "exercise": "_".join(parts[4:])}


def list_recordings(root: Path = MAGCIL_ROOT) -> list[dict]:
    """All recordings with their video and wav paths, sorted by file name."""
    out = []
    for tech in TECHNIQUES.values():
        for video in sorted((root / "videos" / tech).glob("*.mp4")):
            rec = parse_name(video.stem)
            assert rec["technique"] == tech, video
            rec.update(stem=video.stem, video=video, wav=root / "wav" / tech / f"{video.stem}.wav")
            out.append(rec)
    return out
