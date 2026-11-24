# Baseline study — progress log

One section per checkpoint: date, what was run, result, what we learned, where the outputs are.
Plan: [2026-11-23_baseline_study_plan.md](2026-11-23_baseline_study_plan.md). Short status table: `CLAUDE.md` → "Current phase".

**Renumbering (2026-11-24).** The study now also covers GuiFiR-Vid and magcil. New CP7 (GuiFiR preprocessing),
CP8 (magcil preprocessing), CP9 (MMAudio, all datasets), CP10 (V2N, all datasets); the results table is now CP11,
the V2N fine-tune CP12, optional extensions CP13. Sections below keep their original names; "CP7 (part 1)" is GAPS
content of today's CP9/CP11. Plan: "Extension" section in the plan file.

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

---

## CP7 (part 1) — "How bad is MMAudio?" (2026-11-23) — old numbering, now part of CP9/CP11 (GAPS)

Done before CP5/CP6 so the main answer is available early; V2N rows follow after CP6.

**Commands**

```bash
conda activate amt
python scripts/30d_transcribe.py --system mmaudio_noprompt --audio-dir data/interim/baseline_clips/generated/mmaudio_noprompt
python scripts/30d_transcribe.py --system mmaudio_prompt   --audio-dir data/interim/baseline_clips/generated/mmaudio_prompt
conda activate guitar2midi
python scripts/30e_evaluate_baselines.py --run-name cp7_mmaudio \
    --systems real_audio mmaudio_noprompt mmaudio_prompt silence --shuffled real_audio mmaudio_noprompt mmaudio_prompt
```

**New in the evaluation**
- `timing_f50`: pluck-time F1 ignoring pitch (chords merged), so timing can be judged separately from pitch.
- `X@shuffled`: each clip scored against X's output for a clip of a *different piece*. `real_audio@shuffled` =
  "real guitar music, wrong piece" = chance level. Paired Wilcoxon test matched vs shuffled.

**Result** (90 clips, mean; F1 in %)

| system | note onset F1@50 | F1@100 | frame F1 | timing F1 (pitch ignored) | chroma sim | onset corr | notes/clip |
|---|---|---|---|---|---|---|---|
| real audio → transcription (ceiling) | **93.5** | 93.7 | 83.0 | 95.6 | 0.894 | 0.905 | 35.1 |
| MMAudio, video only | **3.3** | 6.1 | 10.7 | 35.9 | 0.198 | 0.114 | 36.8 |
| MMAudio, video + "solo classical guitar" | 3.2 | 5.9 | 10.5 | 35.7 | 0.193 | 0.110 | 36.1 |
| real guitar music, wrong piece (chance) | 2.1 | 4.0 | 9.9 | 28.3 | 0.216 | 0.00 | 35.1 |
| MMAudio, wrong clip | 1.8 | 3.0 | 7.5 | 26.7 | 0.186 | 0.00 | 36.8 |
| silence | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Matched vs shuffled (MMAudio, video only): note F1 3.3 vs 1.8 (p = 7e-4), timing F1 35.9 vs 26.7 (p = 8e-6,
better on 73 % of clips), onset corr 0.11 vs 0.00 (p = 2e-5), chroma 0.198 vs 0.186 (p = 0.45, not significant).

**What we learned**
1. **Notes: almost completely wrong.** 3.3 % note F1 vs 93.5 % possible; only 1–1.5 points above chance.
2. **Pitch / harmony: no information from the video.** Chroma similarity is not better than for a random other
   clip, and even below real guitar music from the wrong piece.
3. **Timing: a weak but real signal.** Pluck timing is clearly above chance (35.9 vs ~27 %, significant), so
   MMAudio reacts to the picking motion, but far from the 95.6 % possible.
4. **Text prompt: no effect.**
5. Note density is right (36.8 vs 34.9 notes per clip): it produces the right *amount* of guitar-like plucking.

**Summary for the supervisor (one paragraph)**

> We ran MMAudio, a state-of-the-art open-domain video-to-audio model, on 90 silent 8-second clips from the 30
> GAPS test performances and transcribed its output with the GAPS benchmark transcription model. On the real
> audio this pipeline reaches 93.5 % note F1, but on MMAudio's audio only 3.3 %, close to the 2 % obtained by
> comparing with a different piece. The generated audio sounds like a plucked guitar with roughly the right
> note density and its pluck timing follows the video slightly better than chance (36 % vs 27 % timing F1),
> but its pitches carry no information about the notes being played (pitch-class similarity equal to a random
> clip). Existing video-to-audio models therefore produce plausible guitar *sound* but not the *music*: they do
> not read the fretboard. This is the gap our visual transcription approach addresses.

**Outputs (local):** `outputs/runs/2026-11-23_cp7_mmaudio/{summary.md, summary.csv, per_clip.csv}`,
transcriptions in `data/interim/baseline_clips/transcriptions/mmaudio_{noprompt,prompt}/`.

---

## CP5 — Fretboard strips for V2N (2026-11-23)

**Goal:** turn each GAPS clip into V2N's input format: a rectified grayscale 800×144 strip of the fretboard
(V2N uses the keyboard), so the piano model can be run zero-shot in CP6. Dataset: the same 90 GAPS test clips;
from V2N only the trained model is used.

**Why tracking was added (decided 2026-11-23).** The plan was one set of clicked corners per clip, i.e. one
fixed crop. That assumes nothing moves, but in GAPS (contact sheet, CP2):
- many videos are edited productions with camera **zoom and pan**,
- players **move the guitar** even with a static camera,
- some clips contain **hard cuts** to another camera angle.
With a fixed crop the fretboard slides out of the strip during the clip, and V2N would partly fail because of
bad cropping instead of because it is a piano model, so the comparison would be unfair. Solution: click once,
then track the corners automatically through the clip. Clicking effort stays the same (one frame per clip),
it covers every kind of movement, and it is a first version of Stage 1 steps 2–4 of the main pipeline
(homography, temporal smoothing, "reuse last good warp"), so it is not throw-away work.

**How it works**
1. `scripts/30f_prepare_strip_frames.py`: hard-cut detection per clip (colour-histogram drop *or* pixel-difference
   jump > 6× the clip's normal motion); the frame to click is the middle of the longest shot.
   Result: **12 of 90 clips have hard cuts** (`027_Zpswc_c3`, `051_rc1wc_c1/c3`, `104_xf1wc_c3`, `222_W41wc_c1/c3`,
   `291_3Sswc_c1/c2/c3`, `354_-k1wc_c1/c2/c3`); in these the longest shot covers 42–85 % of the clip.
   Slow dissolves (e.g. `201_gk1wc_c2`) are not detected as cuts; the tracker quality check flags them.
2. Clicking with the existing tool (`01b_annotate_corners.py`), new: `--no-shuffle` (a video's clips in a row)
   and key `p` = paste the previous clip's clicks (static cameras: 1 s instead of ~10 s per clip).
3. `scripts/30f_make_v2n_strips.py` with `preprocessing/fretboard_tracking.py`: from the clicked frame, forward and
   backward through its shot: features in the neck area (frets, strings, wood) → Lucas-Kanade optical flow with
   forward-backward check → RANSAC homography (ignores the moving hands) → corners moved; if the neck is lost
   (< 15 inliers, invalid quad, > 25 % area jump) the last good corners are held. Corners smoothed over 5 frames.
   Frames outside the clicked shot get a black strip. Warp: nut left, fret 20 right, high-E on top, neck in the
   middle half of the strip height (fingers above/below remain visible).
4. Check: `tracking_sheet.jpg` shows per clip 3 frames with the tracked neck (green = tracked, red = held) and the
   3 strips. Clips with < 90 % tracked frames are flagged. Unusable clips (fretboard not visible) are skipped in
   the tool; V2N and MMAudio are then compared on the same subset.

**Tests:** synthetic textured neck under zoom + pan + rotation is tracked with < 2 px error from a middle frame;
a cut stops the tracking (19 unit tests pass).

**Important correction:** these clicks are on GAPS **test** clips, so they are evaluation-only and must never be
used to train the fretboard estimator (that would leak into the GAPS test evaluation). They are stored in
`data/labels/fretboard_corners/gaps_test_baseline_EVAL_ONLY.jsonl`. (The plan originally said they could be
reused as training data; that was wrong. Only the CP8 clicks on the GAPS train split can.)

**Result (after clicking all 90 frames, fret 12 everywhere)**

```bash
python scripts/30f_make_v2n_strips.py
```

| | clips |
|---|---|
| Strips made | **74** |
| Skipped while clicking (nut or fret 12 not visible: close-ups, wide/low-res shots) | 16: `027_Zpswc_c2`, `031_vpswc_c1`, `100_Lf1wc_c1`, `142_GD1wc_c2`, `201_gk1wc_c1/c2/c3`, `208_c41wc_c2/c3`, `222_W41wc_c3`, `247_sy1wc_c1/c2`, `291_3Sswc_c1/c2`, `354_-k1wc_c2`, `373_4w1wc_c1` |
| Flagged: < 90 % of frames tracked (all because of hard cuts; frames after the cut are black) | 8: `027_Zpswc_c3` 84 %, `051_rc1wc_c1` 67 %, `051_rc1wc_c3` 81 %, `104_xf1wc_c3` 72 %, `222_W41wc_c1` 85 %, `291_3Sswc_c3` 56 %, `354_-k1wc_c1` 63 %, `354_-k1wc_c3` 66 % |
| Frames where the tracker lost the neck | 0 (median 100–390 inlier features per frame) |

**Visual check** (`strips/tracking_sheet.jpg` + close-ups of the lowest-similarity clips):
- Strips are correct in orientation and content: neck horizontal, frets vertical, nut left, soundhole right,
  fingers visible above/below the neck. 640×480 and smaller videos give blurry strips (`019_Vpswc`, `358_441wc`).
- Low strip-to-reference similarity was mostly caused by hands moving over the neck, not by drift.
- **Known failures:** (1) dissolves (gradual cross-fades) are not detected as cuts and the tracker follows the
  blend: first ~2 s wrong in `375_1w1wc_c1` and `341_1M1wc_c2`; (2) slight drift at the strip edges near the end
  of `100_Lf1wc_c3` (neck stays in the strip). These few frames cannot change the zero-shot V2N conclusion; for
  the main pipeline, dissolve detection and an appearance check against the reference frame are needed.

**Lesson for the main pipeline:** with 16/90 clips unusable (no full fretboard in view) and 8/90 with cuts,
~18 % of GAPS test clip time has no usable fretboard view; the final model must handle missing/partial views
(e.g. hold or mask), and the GAPS visibility audit should report this per piece.

**Outputs (local):** `data/interim/baseline_clips/strips/{<clip>.npy, strips.csv, tracking_sheet.jpg}`,
clicks `data/labels/fretboard_corners/gaps_test_baseline_EVAL_ONLY.jsonl`.

### CP5 addendum — masked evaluation (2026-11-23)

**Problem:** in 8 clips with camera cuts the strips are black outside the clicked shot, and in 2 clips a
dissolve gives a wrong view for the first frames. V2N cannot see anything there, so counting the notes in
those parts as V2N errors would be unfair (our cropping's fault, not V2N's).

**Decision:** evaluate only the time with a valid fretboard view, for *every* system (also MMAudio and the
ceiling), so all rows of the table use exactly the same clips and frames. Chosen over clicking extra frames in
the other shots (~10 more clicks) because it needs no extra annotation and the lost time is small.

**How**
- `30f_make_v2n_strips.py` writes `strips/<clip>_valid.npy` (per 25 fps frame): inside the clicked shot and not
  listed in `data/interim/baseline_clips/invalid_frames.txt`. Invalid frames get a black strip.
- `invalid_frames.txt` (checked by eye on the strips): `375_1w1wc_c1 0–64` and `341_1M1wc_c2 0–64` (dissolves).
- `note_metrics.evaluate(..., valid=mask)`: only notes with a valid onset frame are scored; frame F1, chroma and
  onset correlation use valid frames only (unit-tested).
- `30e_evaluate_baselines.py --mask-dir data/interim/baseline_clips/strips`: only the 74 clips with strips.

**Masked subset:** 74 clips, 96.1 % of their frames valid; 10 clips flagged (< 90 % valid: 8 cuts + 2 dissolves).

**MMAudio on the masked subset** (run `2026-11-23_cp7_mmaudio_masked`) — same picture as on all 90 clips:

| system | note F1@50 | timing F1 | chroma sim | onset corr |
|---|---|---|---|---|
| real audio (ceiling) | 93.6 | 95.4 | 0.892 | 0.903 |
| MMAudio, video only | 3.1 | 35.5 | 0.205 | 0.111 |
| MMAudio + prompt | 3.0 | 35.7 | 0.203 | 0.105 |
| real music, wrong piece (chance) | 2.1 | 28.1 | 0.213 | −0.01 |

**What "masked subset" means:** (1) only the 74 clips with clicked corners (the 16 skipped clips are left out),
and (2) inside those clips only the frames where the tracked box shows the correct fretboard: the clicked camera
shot, minus the dissolve frames. Example: `291_3Sswc_c3` is valid from frame 88 to 200, so a GT note at 1.0 s
(frame 25) is ignored and one at 5.0 s (frame 125) is scored, for every system.

**Why MMAudio was re-evaluated on it, and why the numbers hardly change**
- Purpose: a *representativeness check*. Dropping 16 clips and some frames could have removed especially easy or
  hard material and shifted the results. All 90 vs masked 74: ceiling 93.5 → 93.6, MMAudio 3.3 → 3.1, chance
  2.1 → 2.1. Nearly identical, so the subset is representative of the GAPS test clips and results on it are
  comparable with the full-set results.
- The mask hardly affects MMAudio by design: MMAudio always sees the full original video (not the strips), and
  its score is already near chance. The mask exists for **V2N**, which only sees the strips and would otherwise
  be charged for notes in black (no-view) frames.

From now on the masked subset is the reference for all rows that include V2N.

## CP6 — V2N zero-shot on the guitar strips (2026-11-24)

**Setup.** Public checkpoint `v2n_pianovam.safetensors` (release v0.1, trained on PianoVAM), V2N repo at commit
`a6408c4` in `~/external/V2N`, env `v2n` (`envs/v2n.yml`). Wrapper `guitar2midi.baselines.v2n` uses V2N's own
model, window stitching and note decoder unchanged; only the input loading is ours (our 800×144 `.npy` strips instead
of their HDF5). Same settings as V2N's test time: 1 s windows, stride 13 frames, centre stitching, onset/frame
threshold 0.5, fp32. Script `scripts/30g_run_v2n.py` → `transcriptions/v2n_zeroshot/`, probabilities in
`v2n_probs/`, plots in `plots/v2n/`. 74 clips in 28 s.

**Positive control first (is our wrapper right?).** `scripts/30g_check_v2n_piano.py` runs the wrapper on one
PianoVAM test video (`2024-02-17_21-44-37`, 178 s), preprocessed with V2N's own warp function. Result
(`outputs/runs/2026-11-24_cp6_v2n_piano_check/`): 538 notes, identical to V2N's shipped prediction for that file
(onset F1 ours vs shipped = **100 %**), and 94.3 % onset F1 vs GT (paper, whole test set: 94.7 %). So the wrapper
reproduces V2N exactly, and whatever it does on guitar is V2N's behaviour, not a bug in our input path.

**Result on guitar** (run `2026-11-24_cp7_v2n_masked`, masked subset, 74 clips):

| system | note F1@50 | timing F1 | frame F1 | chroma sim | onset corr | notes/clip |
|---|---|---|---|---|---|---|
| real audio (ceiling) | 93.6 | 95.4 | 82.8 | 0.892 | 0.903 | 32.8 |
| MMAudio, video only | 3.1 | 35.5 | 10.3 | 0.205 | 0.111 | 33.6 |
| **V2N zero-shot** | **0.0** | **0.0** | **0.0** | 0.000 | −0.001 | **0.0** |
| silence | 0.0 | 0.0 | 0.0 | 0.000 | 0.000 | 0.0 |

V2N predicts **one note in 74 clips** (73 clips empty), so on every metric it equals the silence baseline.

**Is there a signal below the threshold?** No. The onset probabilities stay around 1e-3 (max 0.50 in one clip),
as static horizontal bands at low piano pitches that do not change when notes are played (see the log-scale panel
in `plots/v2n/*.png`). Threshold-free check: correlation of V2N's per-frame max onset probability with GT onset
times is r = −0.026 for the matched clip vs −0.022 for a different piece (matched better on 47 % of clips,
Wilcoxon p = 0.67). So V2N carries no timing information about guitar, and lowering the threshold would only add
random notes.

**Why (interpretation).** V2N was trained on one fixed top-down camera over one piano. The S2S slope prior and
all heads assume a keyboard: white/black key pattern, fingers pressing down into the keys, a 100 px margin below the
keys. A fretboard strip (frets as thin vertical lines, strings horizontal, the fretting hand often from above, motion
blur) is far out of its training distribution, and the model answers "no key pressed". This matches V2N's own
cross-dataset result (transfer collapses when the geometry differs). Nut-left orientation was a deliberate choice
(low pitches left, as on the keyboard); other orientations were not tried, because with no timing signal at all a
flip cannot create one.

**Takeaway for the thesis.** Neither existing route works on guitar video: V2A (MMAudio) gets rhythm weakly
(timing F1 35.5 vs chance ~27) but almost no pitch (note F1 3.1 vs chance 2.1); the closest visual transcription
model (V2N) outputs silence. This motivates CP8 (fine-tune V2N on guitar) and the guitar-specific pipeline.

**Open (Luckey, 2026-11-24): the 0.0 result does not look right yet — to be re-examined before CP7 is
finalised.** Things to check then (none done yet):
- Strip geometry vs V2N's piano crop: V2N's crops include a 100 px margin below the keys (hands come from there);
  ours put the neck in the middle 50 % of the height. Try a crop/orientation closer to the piano layout
  (e.g. horizontal flip, vertical flip, neck in the top part) on a few clips and see if the probabilities move at all.
- Check the input statistics (brightness/contrast) of our strips against PianoVAM strips.
- Lower decoding thresholds only as a diagnostic (not for the table), to see whether any timing structure appears.
- If V2N still stays silent, keep 0.0 as the zero-shot result; the positive control already rules out a wrapper bug.

Next step after that: CP8 draft plan in [2026-11-24_v2n_guitar_adaptation_plan.md](2026-11-24_v2n_guitar_adaptation_plan.md)
(train the V2N architecture on GAPS guitar strips; the released V2N uses no pretrained foundation model — its
ResNet-18 is trained from scratch on piano — but the code has an ImageNet-22k ConvNeXt-T option).
