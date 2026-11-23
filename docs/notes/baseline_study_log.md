# Baseline study — progress log

One section per checkpoint: date, what was run, result, what we learned, where the outputs are.
Plan: [2026-11-23_baseline_study_plan.md](2026-11-23_baseline_study_plan.md). Short status table: `CLAUDE.md` → "Current phase".

Note: `data/` and `outputs/` are not in git. Paths below are local; rerun the command to regenerate them.

---

## CP0 — Plan saved (2026-11-23)

- Plan written to `CLAUDE.md` and `docs/notes/2026-11-23_baseline_study_plan.md`. Commit `c598b03`.
- Context: supervisor meeting 2026-11-23. FPE / GuiFiR work paused; first measure how bad existing video→audio / video→MIDI models are on guitar.

---

## CP1 — GAPS data in place (2026-11-23)

**Command**

```bash
python scripts/30a_download_gaps.py --split test
```

**Source:** Hugging Face `xavriley/GAPS` v1.1 (metadata, audio, MIDI) + YouTube via `yt_id` (videos, max 720p, with audio track).
Not the v1 Zenodo zip in `~/Downloads/gaps_v1_no_audio.zip`: its file names clash by case (`bc1wc` / `Bc1wc`) and it has no audio.

**Result**

| | Test split |
|---|---|
| Pieces | 30 (27 performers) |
| MIDI | 30/30 |
| Audio | 30/30, 1.77 h, 1.2 GB |
| Videos | 30/30, none missing, 853 MB |
| Usable (audio + video) | 1.77 h |

Per piece (audio/video length in s):

| id | audio | video | fps | resolution |
|---|---:|---:|---:|---|
| 019_Vpswc | 156.1 | 156.2 | 29.97 | 640x480 |
| 027_Zpswc | 404.9 | 405.0 | 23.976 | 1280x720 |
| 031_vpswc | 186.5 | 186.6 | 25 | 1280x720 |
| 043_bc1wc | 328.0 | 328.1 | 29.97 | 1280x720 |
| 051_rc1wc | 146.1 | 146.1 | 23.976 | 1280x720 |
| 063_bV1wc | 181.5 | 181.5 | 30 | 1280x720 |
| 100_Lf1wc | 222.6 | 222.7 | 25 | 1280x720 |
| 104_xf1wc | 184.4 | 184.5 | 29.97 | 1280x720 |
| 111_hf1wc | 250.3 | 250.4 | 29.97 | 1280x720 |
| 112_mf1wc | 326.7 | 326.7 | 29.97 | 1280x720 |
| 118_VD1wc | 120.7 | 120.8 | 25 | 1280x720 |
| 126_XD1wc | 182.6 | 182.7 | 29.97 | 1280x720 |
| 142_GD1wc | 238.9 | 238.9 | 23.976 | 1280x720 |
| 179_pM1wc | 189.1 | 189.2 | 23.976 | 1280x720 |
| 201_gk1wc | 190.9 | 191.0 | 25 | **320x240** |
| 208_c41wc | 200.2 | 200.3 | 29.97 | 1280x720 |
| 212_y41wc | 322.5 | 322.5 | 25 | 640x480 |
| 222_W41wc | 148.5 | 148.5 | 23.976 | 1280x720 |
| 235_Ny1wc | 415.8 | 415.9 | 30 | 848x480 |
| 247_sy1wc | 131.8 | 131.9 | 29.97 | 1280x720 |
| 263_cw1wc | 296.9 | 296.9 | 25 | 1280x720 |
| 270_Jw1wc | 102.7 | 102.7 | 30 | 1280x720 |
| 291_3Sswc | 134.9 | 135.0 | 29.97 | 1280x720 |
| 294_BSswc | 125.0 | 125.0 | 24 | 1280x720 |
| 303_5N1wc | 154.6 | 154.7 | 30 | 1280x720 |
| 341_1M1wc | 127.4 | 127.5 | 23.976 | 1280x720 |
| 354_-k1wc | 127.1 | 127.2 | 25 | 1280x720 |
| 358_441wc | 134.4 | 134.4 | 29.97 | 640x480 |
| 373_4w1wc | 428.2 | 428.3 | 25 | 1280x720 |
| 375_1w1wc | 207.4 | 207.5 | 25 | 640x360 |

**What we learned**

- All test videos are still on YouTube.
- Audio and video lengths match within 0.1 s for every piece, so GAPS audio is most likely the video's own audio track. Sync is checked properly in CP2.
- Frame rates vary (23.976–30 fps) → resample to 25 fps as in V2N.
- One very low-resolution video (`201_gk1wc`, 320x240); possibly unusable for V2N strips.
- yt-dlp warns that Python 3.10 is deprecated; works for now, a future yt-dlp may need 3.11.

**Outputs (local):** `data/raw/gaps/{audio,midi,videos}/`, `data/raw/gaps/summary_test.csv`, `data/raw/gaps/videos/failed_test.tsv` (empty).
