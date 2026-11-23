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

---

## CP2 — Evaluation clips (2026-11-23)

**Command**

```bash
python scripts/30b_make_eval_clips.py
```

**What it does:** for every GAPS test piece, 3 non-overlapping 8 s windows (one per third of the piece, random among
windows with ≥ 8 note onsets, seeded). Per clip: silent video resampled to 25 fps, real GAPS audio, GT MIDI (notes
with onset in the window, shifted to 0, offsets clipped at the window end). Changed from the plan: all 30 pieces
instead of ~20, so pieces without a visible guitar can be removed and enough remain.

**Result**

| | |
|---|---|
| Clips | 90 (30 pieces × 3), 12.0 min |
| Notes per clip | median 31 (range 12–77) |
| Sync correlation | 0.98–1.00 for every clip |
| Sync lag (video audio vs GAPS audio) | constant per piece: 0 ms (2 pieces), 13.1 ms (2), 23.2 ms (1), 36.3 ms (21), 49.4 ms (4) |
| Residual lag after correction | < 0.1 ms on all 90 clips |

**Sync finding.** GAPS audio (and therefore its MIDI) is *earlier* than the picture by a constant amount per piece.
The values are exactly audio-codec priming delays at 44.1 kHz: 576 samples = 13.1 ms (MP3), 1024 = 23.2 ms (AAC),
1600 = 36.3 ms, 2176 = 49.4 ms (sums of these). So the GAPS wav was decoded from a differently encoded copy of the
YouTube audio. Instead of dropping the 4 pieces with > 40 ms (1 frame) lag, as first planned, every clip is corrected:
video window [t, t+8) is paired with GAPS audio/MIDI [t − lag, t − lag + 8). Without this, most GT onsets would be
~1 frame early relative to the video. **This matters for the whole thesis, not only the baseline: any GAPS
video↔MIDI pairing must apply this per-piece lag.**

**Contact sheet** — first version showed only the middle frame per clip; now 3 frames per clip (start, middle,
end; `--frames-per-clip`), which also reveals camera cuts and non-guitar shots inside a clip.

**Clips without a guitar and how they were fixed (goal: keep all 90 clips)**
- First sheet: no guitar in `358_441wc_c1` (album cover), `358_441wc_c3` (landscape), `208_c41wc_c1` (Christmas decoration).
- Step 1, `--reject <clip_id>`: marks the window as bad (`rejected_windows.txt`) and draws a new random window from the
  same part of the piece. Only the rejected clips change. Did not help here: the new windows were again landscapes / a dissolve.
- Step 2, timeline check (one frame every 5 s of the full video): `358_441wc` is a slideshow-style video, the guitarist
  is only visible at ~25–58 s and ~75–92 s; `208_c41wc` is clear at ~30–60 s (dissolves around 20–25 s and 65 s).
- Step 3, `--set <clip_id>=<start>` (`manual_windows.txt`, column `window=manual` in clips.csv):
  `358_441wc_c1=27`, `358_441wc_c2=42`, `358_441wc_c3=77`, `208_c41wc_c1=31`. All frames now show the guitarist.
- Lesson for CP8 (270 train videos): some GAPS videos contain long non-guitar parts, so a per-video visibility check
  is needed there (automatic, e.g. CLIP zero-shot, instead of by eye).

**Other observations**
- Many videos are edited multi-camera productions (cuts between wide shot and close-ups inside a piece, e.g. `100_Lf1wc`, `201_gk1wc`, `291_3Sswc`, `373_4w1wc`). Fine for V2A; for V2N strips (CP5) one homography per clip only works if there is no cut inside the clip.
- `201_gk1wc` is 320×240.
- Camera views range from full body to fretboard close-ups; mostly front-facing, nut on the image right.

**Final:** 90 clips from 30 pieces, 12.0 min; 86 random windows, 4 manual.

**Outputs (local):** `data/interim/baseline_clips/{clips.csv, contact_sheet.jpg, rejected_windows.txt, manual_windows.txt, video/, audio/, midi/, plots/}`
(plots: `019_Vpswc_c1`, `118_VD1wc_c1`, `263_cw1wc_c1`).

---

## CP3 — Transcription ceiling (2026-11-23)

**Why:** V2A models output audio, so their notes are measured by transcribing that audio. The transcription model
makes mistakes too; running it on the *real* audio gives the best score possible (ceiling). Any gap between the
ceiling and a V2A system is the V2A model's fault. It also checks that audio ↔ MIDI alignment of our clips is right.

**Commands**

```bash
conda activate amt                     # separate env: conda env create -f envs/amt.yml; pip install -e . --no-deps
python scripts/30d_transcribe.py --system real_audio
conda activate guitar2midi
python scripts/30e_evaluate_baselines.py --run-name cp3_ceiling
```

**Model:** GAPS benchmark model (Riley et al. 2024; Kong et al. high-resolution CRNN), code
github.com/xavriley/hf_midi_transcription @ `96f6797`, weights HF `xavriley/midi-transcription-models`. Own env
`amt` because the package needs a recent torch (2.15 installed), which would break mmcv in `guitar2midi`.
Two GAPS checkpoints compared; `guitar-gaps-paper-version-12200_iterations.pth` chosen and used for all systems.

**Metrics** (`src/guitar2midi/evaluation/note_metrics.py`, unit-tested; mean over 90 clips):
onset P/R/F1 at 50 and 100 ms (mir_eval, onsets only; main metric as in GAPS and V2N), onset+offset F1, frame F1
at 25 fps, chroma similarity (pitch-class agreement per frame), onset correlation (rhythm agreement, ignores pitch).

**Result**

| system | onset F1@50 | P | R | F1@100 | onset+offset F1 | frame F1 | chroma sim | onset corr | notes/clip |
|---|---|---|---|---|---|---|---|---|---|
| real audio, paper ckpt (**ceiling**) | **93.5** | 93.5 | 93.8 | 93.7 | 63.6 | 83.0 | 0.894 | 0.905 | 35.1 |
| real audio, default ckpt | 93.5 | 94.3 | 93.0 | 93.6 | 56.4 | 80.8 | 0.872 | 0.910 | 34.3 |
| silence | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

GT: 34.9 notes per clip on average.

**What we learned**
- Ceiling onset F1 93.5 % matches the GAPS paper (Table 6: 93.4–94.3 %) → model, clip cutting and audio/MIDI alignment are correct.
- Per clip: median 95.0 %, worst 67.7 % (`051_rc1wc_c3`); no broken clip.
- Onset+offset F1 is much lower (64 %): guitar note ends are ambiguous even from audio (relevant for the offset-head question).
- Transcription of 90 clips takes ~15 s on the RTX 4080.

**Outputs:** `data/interim/baseline_clips/transcriptions/{real_audio, real_audio_defaultckpt}/` (local),
`outputs/runs/2026-11-23_cp3_ceiling/{per_clip.csv, summary.csv, summary.md, args.json}` (local).

---

## CP4 — MMAudio on the guitar clips (2026-11-23)

**Commands**

```bash
conda activate mmaudio                 # conda env create -f envs/mmaudio.yml; pip install -e . --no-deps
python scripts/30c_run_mmaudio.py
conda activate guitar2midi
python scripts/30c_plot_spectrograms.py
```

**Model:** MMAudio `large_44k_v2` (Cheng et al., CVPR 2025), code @ `974010a`, settings of the official demo
(25 Euler steps, CFG 4.5, bf16), seed 42 for every clip. Own env `mmaudio` (torch 2.15). Weights 5.9 GB in
`~/.cache/mmaudio` (+ BigVGAN in the HF cache).

**Systems**
- `mmaudio_noprompt`: silent video only.
- `mmaudio_prompt`: silent video + text "solo classical guitar".

**Result**
- 90 clips × 2 systems generated in ~9 min (~3 s per clip on the RTX 4080).
- Listening examples (silent clip + real / generated audio): `data/interim/baseline_clips/examples/<clip>__{real,mmaudio_noprompt,mmaudio_prompt}.mp4`
  for `019_Vpswc_c1`, `100_Lf1wc_c1`, `142_GD1wc_c1`, `235_Ny1wc_c1`, `303_5N1wc_c1`.
- Spectrograms real vs generated with GT notes overlaid: `examples/<clip>__spectrograms.png`.

**What we see (spectrograms, before any numbers)**
- Generated audio looks like plucked strings: sharp attacks and harmonic overtones, not noise or speech.
- Rhythm is partly right: in `100_Lf1wc_c1` the strong attacks at ~2.0, 3.0, 5.1, 6.2 s match the real recording,
  so MMAudio picks up the picking motion from the video.
- Pitches mostly do not follow the GT notes. The prompt changes little.
- To be quantified in CP7 (onset F1, chroma similarity, onset correlation).

**Leakage checks**
- 0/90 silent clips contain an audio stream.
- 0/300 GAPS YouTube IDs (train + test) are in VGGSound (MMAudio's audio-visual training set), so MMAudio has not
  seen these performances with their sound. (MMAudio's audio-text data, e.g. AudioSet captions, has no video conditioning.)

**Caveats**
- MMAudio reads our 8.00 s clips as 7.96 s (one frame), so the last 40 ms have no generated audio (same for every clip, negligible).
- MMAudio's sync encoder center-crops frames to a square: on wide shots the hands can be partly outside it.
