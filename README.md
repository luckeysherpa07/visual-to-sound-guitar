# visual-to-sound-guitar

Master's thesis (JMU Würzburg): predicting MIDI notes from silent guitar video by adapting the visual piano transcription pipelines V2N and Sight-to-Sound. Frames are rectified into a fixed fretboard strip, then a V2N-style model predicts onsets and held notes. See `CLAUDE.md` for the full plan.

## Setup

```bash
conda env create -f environment.yml
conda activate guitar2midi
bash scripts/setup_mmpose.sh   # pinned mmcv / mmdet / mmpose
pip install -e .[data]         # only for downloading GuiFiR (gdown, yt-dlp)
```

## Layout

- `src/guitar2midi/` — importable code (data, preprocessing, model, training, decoding, evaluation)
- `scripts/` — numbered CLI entry points, one per pipeline step
- `configs/` — YAML configs per task
- `data/`, `outputs/` — local only, not in git
- `notebooks/` — exploration
- `thesis/` — LaTeX source
- `docs/papers/` — reference PDFs
