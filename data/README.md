# data/ (not in git)

Each pipeline stage reads one folder and writes the next.

| Folder | Contents |
|---|---|
| `raw/{gaps,magcil,own}/` | Untouched downloads and recordings. Never modify. |
| `interim/landmarks/` | Per-frame fretboard landmarks. |
| `interim/homographies/` | Smoothed per-frame homographies. |
| `processed/strips/` | Cached warped fretboard strips (~800x144 gray, 25 fps). Never recomputed in training. |
| `labels/` | Note/frame labels plus metadata. Every entry has `label_source: real` or `pseudo`. |
| `splits/` | train/val/test lists, split by performer and recording. |
