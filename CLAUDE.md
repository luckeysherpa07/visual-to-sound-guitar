# CLAUDE.md — Guitar Video → MIDI Thesis

Master's thesis (Computer Science, JMU Würzburg). Author: Ruluckey "Luckey" Sherpa. Supervisor: Zongwei Wu.
**Hard deadline: Feb 1, 2027. Experiment freeze: Jan 6, 2027.**
Status as of 2026-11-24: CP0–CP6 done (CP6: V2N zero-shot = silence on guitar). Baseline study extended to GuiFiR-Vid and magcil (new CP7–CP10, checkpoints renumbered; results table is now CP11). CP7 (GuiFiR preprocessing) done; CP8 (magcil preprocessing), CP9 (MMAudio on all datasets) and CP10 (V2N on all datasets, 0.0 re-checked) done; CP11 (results table + supervisor summary) done; CP12 (full-string strips nut → bridge) done; next: supervisor review, then CP13 (V2N fine-tuning pilot, runs A–D). After the supervisor meeting on Nov 23, the FPE work (Stage 1) is **paused** and the current phase is the **baseline study** (see below): measure how badly existing off-the-shelf models do on our guitar data before building our own. After that, GuiFiR work resumes, now including the hand pose estimator. FPE state when paused: code written and verified end to end on fake fretboards, GuiFiR-Vid prepared (val/test COCO files), corner annotation tool finished, 3 train frames clicked, no real training run. Update this file when decisions change.

## Goal

Predict MIDI notes from **silent video** of guitar playing (visual music transcription). The approach adapts the visual piano transcription pipelines V2N and Sight-to-Sound (S2S) to guitar.

Scope is locked to MIDI prediction. Audio generation and performance feedback (earlier ideas) are out of scope.

## Input / output

- **Input:** guitar performance video, no audio at inference. Frames are preprocessed into a rectified, fixed-size, grayscale fretboard strip (target ~800×144, as in V2N), in 1 s clips at 25 fps.
- **Output:** note events (pitch, onset, and offset where reliable) written as MIDI.
- **Not predicted:** velocity. GAPS velocities are not real, so there is no trustworthy supervision.

## Current phase: baseline study (existing models) — started 2026-11-23

Supervisor's request: before going into GuiFiR detail, get a bird's-eye view of how well existing video→audio / video→MIDI models already do on guitar ("how bad are they really"). Full plan: `docs/notes/2026-11-23_baseline_study_plan.md`. Progress log (results of every checkpoint, update after each CP): `docs/notes/baseline_study_log.md`.

- **Core idea:** silent video → V2A model → generated audio → audio transcription → MIDI → compare with GAPS ground-truth MIDI (mir_eval note F1). The same transcription model on the **real** audio is the ceiling, so the gap is the V2A error.
- **Models:** MMAudio (primary V2A, open weights, fits the RTX 4080 16 GB); V2N zero-shot (piano video→MIDI on warped guitar strips); optional HunyuanVideo-Foley / ThinkSound / Foley Music. No public guitar video→MIDI model exists.
- **Data:** GAPS test split (real MIDI; 30 pieces × 3 clips × 8 s), plus (added Nov 24) GuiFiR-Vid test split (≤ 32 clips, 1 per song) and magcil (27 clips, 3 per technique), both scored against the GAPS model's transcription of their real audio (pseudo-GT); own data later with the same scripts.
- **Checkpoint rule (Nov 24):** one checkpoint per dataset for all of its preprocessing (clips, reference transcription, strips), then one checkpoint per task that runs on all datasets. Old CP7/8/9 became CP11/12/13; earlier names (log "CP7 (part 1)", run `2026-11-24_cp7_v2n_masked`) are kept as they are. Later the same day the full-string strips were inserted as CP12, so the V2N fine-tune is CP13 and the optional extensions CP14.
- **External models run in their own conda envs** (`mmaudio`, `v2n`; they need newer torch than our pinned 2.1.2). They exchange data with our code only through files (clip CSV, wav, MIDI).
- Each checkpoint ends with a visible output and a stop for review.

| CP | What | You see | Status |
|---|---|---|---|
| 0 | Save plan in CLAUDE.md + `docs/notes/` | CLAUDE.md diff | done |
| 1 | GAPS data: metadata, audio, MIDI from HF `xavriley/GAPS` v1.1 (not the v1 zip: case-clashing names), test videos (yt-dlp) | summary: pieces, videos ok/missing, hours | done (Nov 23): test 30/30 pieces, 27 performers, 1.77 h, all videos available; audio and video lengths match within 0.1 s; fps 23.976–30, mostly 720p (one 320×240: `201_gk1wc`) |
| 2 | Eval clips: silent video + real audio + GT MIDI per clip, sync check | `clips.csv`, contact sheet, 3 piano rolls | done (Nov 23): 90 clips (30 pieces × 3 × 8 s, 25 fps), median 31 notes/clip. GAPS audio/MIDI lag the video's audio by a constant 0–49 ms per piece (codec priming); corrected, residual < 0.1 ms. Contact sheet 3 frames/clip; no-guitar clips fixed (`--reject` redraws, `--set` manual windows for slideshow-style `358_441wc` and `208_c41wc_c1`), still 90 clips |
| 3 | Transcription ceiling on real audio (GAPS authors' model) + metrics code | ceiling F1 ≈ GAPS paper level | done (Nov 23): onset F1@50 ms **93.5 %** (paper: 93.4–94.3), frame F1 83.0, median per clip 95.0, min 67.7. Checkpoint `guitar-gaps-paper-version-12200_iterations.pth` (env `amt`), used for all systems |
| 4 | MMAudio on all clips, no prompt and "solo classical guitar" | ~5 videos with generated audio | done (Nov 23): large_44k_v2, seed 42, 90 × 2 clips in ~9 min (env `mmaudio`). Output looks like plucked strings with plausible rhythm, pitches mostly off (spectrograms). No leakage: clips have no audio stream; 0/300 GAPS videos in VGGSound |
| 5 | Fretboard strips for V2N: cut detection, click nut + fret 12 on one frame per clip, **track the corners through the clip**, warp to 800×144 strips | tracking sheet (3 frames + strips per clip) | done (Nov 23): 74 clips with strips, 16 skipped (fretboard not fully visible). Frames outside the clicked shot (cuts) or in dissolves (`invalid_frames.txt`) are masked: `<clip>_valid.npy`; evaluation with `30e --mask-dir` scores all systems on the same 74 clips / 96 % of frames (MMAudio there: 3.1 % note F1, ceiling 93.6 %) |
| 6 | V2N zero-shot on the strips (public checkpoints) | predicted vs GT piano roll | done (Nov 24): wrapper verified on a PianoVAM test video (identical to V2N's shipped prediction, 94.3 % onset F1). On guitar: **1 note in 74 clips** = silence baseline (all metrics 0); onset probs ~1e-3, no timing signal (r with GT onsets = shuffled, p = 0.67). Re-checked in CP10 (16 strip variants, 3 datasets): 0.0 confirmed |
| 7 | **GuiFiR-Vid preprocessing:** eval clips from the test split (1 per song, ≤ 32, 8 s, 25 fps, audio from the mp4), reference transcription of the real audio, tab check (`frame_tab` + capo → pitch at tab frames; how often the reference has it), V2N strips seeded from GuiFiR GT corners (no clicking; test corners = eval only) | clips.csv, contact sheet, tab-agreement %, tracking sheet | done (Nov 24): 32 clips (1 per song, 4.3 min, ≤ 1280 px), no cuts; strips 100 % coverage, tracking error vs GuiFiR GT corners median 1.3 px (640×360). Test tab is **manual** (GuiFiR paper); `frame_tab[0]` = high E, pitch = open + capo + fret. Reference finds **69.9 %** of tab pitches as onsets (±100 ms; per clip median 79.6 %), 83 % sounding; 8 clips < 50 % (merged repeated notes, a few offset/tuning mismatches) |
| 8 | **magcil preprocessing:** extract `~/Downloads/guitar_style_dataset-v1.0.0.zip` into `data/raw/magcil/`, check view/fps/video↔wav sync, eval clips (3 per technique, 27), reference transcription + spot check (electric guitar vs classical-trained model), V2N strips from corner clicks | summary, contact sheet, piano rolls, tracking sheet | done (Nov 24): data extracted (549 recordings, 1 player, smartphone 1080p30; some `rotate=180`, some mirrored); **27 clips** (3 per technique × guitar/amp pairs, 8 s or whole recording if shorter: `length_s` column; bend 3.2–3.4 s), sync ok (−21 ms AAC priming corrected on 4). Reference: median 63 notes/clip; 98 % of notes in the exercise scores' pitch classes (chance 58 %); misses slid notes on slide clips. Corners clicked (`magcil_baseline_EVAL_ONLY.jsonl`, 4 skipped) → **23 clips with strips**, 100 % coverage |
| 9 | **MMAudio on all datasets** (± prompt), transcribe, evaluate vs GT/reference (GAPS outputs from CP4 reused) | spectrograms, rows per dataset | done (Nov 24): prompts "solo classical/acoustic/electric guitar" per set. Note F1@50 (masked): GAPS 3.1 / 3.0 (chance 2.1), **GuiFiR 2.7 / 2.8 (chance 1.6), magcil 5.9 / 6.6 (chance 4.3)**; timing F1 above chance only on GAPS (35.5 vs 25.8 shuffled, p 4e-5), not on GuiFiR (p 0.66) or magcil (p 0.82); GuiFiR tab recall 6.1 / 5.3 % vs reference 69.9 %, chance 3.0 %. Runs `2026-11-24_cp9_mmaudio_{gaps,guifir,magcil}_masked` |
| 10 | **V2N zero-shot on all datasets** + the open CP6 re-check (flip / crop layout / thresholds) | rolls, rows per dataset | done (Nov 24): **0 notes** on GuiFiR (32 clips) and magcil (23) = silence, as on GAPS. Re-check: 4 flips × 2 layouts (incl. piano-like neck-on-top) × brightness matched to piano strips (ours darker: mean ~100 vs 155) = 16 variants × 3 datasets: no notes, onset F1 0.0 even at threshold 0.05, no timing/pitch correlation above shuffled. 0.0 confirmed. Runs `2026-11-24_cp10_v2n_recheck`, `…_cp10_strip_stats`, `…_cp10_v2n_{guifir,magcil}_masked` |
| 11 | **Results table** (was CP7): one table, block per dataset, label-source column (real MIDI / pseudo-GT), GuiFiR tab-recall column; ceiling / MMAudio ± prompt / V2N zero-shot / shuffled video / silence | table + summary for supervisor | done (Nov 24): `scripts/30i_results_table.py` → `thesis/tables/baseline_results.tex` (+ md/csv in `outputs/runs/2026-11-24_cp11_results_table/`), from runs `2026-11-24_cp11_{gaps,guifir,magcil}`. Note F1@50: real audio 93.6 (GAPS); MMAudio 3.1 / 2.7 / 5.9 (GAPS / GuiFiR / magcil) vs chance 2.1 / 1.6 / 4.3; V2N 0.0 everywhere; MMAudio beats a mismatched video only on GAPS (timing 35.5 vs 25.8). Supervisor summary: `docs/notes/2026-11-24_baseline_summary_for_supervisor.md` (not sent yet) |
| 12 | **Full-string strips (all datasets):** second strip version nut → bridge (bridge = 2 × nut–fret-12 distance, from the same tracked corners; no new clicks), so the **picking hand** is in the strip, not only the fretting hand; for the test clips of GAPS / GuiFiR-Vid / magcil; tracking sheet check (bridge part is extrapolated, may be out of frame); V2N zero-shot (piano model, no training) on the nut → bridge strips as one more variant (expected silent; the real picking-hand test is CP13 run D). Same valid-frame masks as before | tracking sheet with both strip versions, V2N zero-shot rows on the nut → bridge strips | done (Nov 24): corners per frame now saved (`strips/<clip>_corners.npy`, strips byte-identical after rerun); `fretboard_grid.extend_corners` (bridge = 1.46 × nut–fret-20) → `strips_full/` via `scripts/30j_make_full_strips.py`. Geometry checked (GuiFiR classical: strip end on the bridge saddle); picking hand visible in wide/medium shots, black where the body is out of frame (black share median GAPS 5 %, GuiFiR 0 %, magcil 21 %). V2N zero-shot on them: 0 notes on all three sets. Runs `2026-11-24_cp12_v2n_full_*` |
| 13 | V2N fine-tuning pilot (was CP8, then CP12): guitar heads, GAPS train strips (1 click set per video); full fine-tune vs scratch (LoRA optional); evaluated on all three baseline sets (GAPS = main, GuiFiR-Vid + magcil = unseen cross-dataset, pseudo-GT + GuiFiR tab recall) | curves + table rows per dataset | todo. Draft plan (not approved): `docs/notes/2026-11-24_v2n_guitar_adaptation_plan.md` — no V2N model-code change needed: `Transcriber_OnsetFrame`, 88 outputs kept, our strips/labels written in V2N's HDF5/TSV format; runs A piano-init fine-tune, B ConvNeXt-T ImageNet-22k, C scratch, D = A on full-string strips (picking hand visible); trained on GAPS only |
| 14 | Optional (was CP9, then CP13): more V2A models, own data (own data = one more preprocessing CP) | extra table rows | todo |

Targets: CP7–CP11 by ~Nov 30, CP12 by ~Dec 1, CP13 by ~Dec 6. If CP13 has no results after ~4 days, move it to the main experiments phase.

## Pipeline

```
video ─► fretboard landmarks ─► homography ─► temporal smoothing ─► warped strip
      ─► V2N-style model ─► onset / frame (hold) [/ offset] heads ─► peak decoding ─► MIDI
```

### Stage 1: fretboard preprocessing (paused 2026-11-23; resumes after the baseline study, with hand pose)

1. **Landmark detection per frame.** Method: GuiFiR fretboard pose estimator (FPE) only, no hand pose / TabCNN / joint fine-tuning.
   - Model (GuiFiR §IV-A.2): MMPose top-down regression, ResNet-50 (ImageNet init) → global average pooling → linear → 126 (x, y). Input = fretboard box + 10% padding, 256×256 crop.
   - 126 points = 6 strings × (nut + 20 frets); index `string * 21 + fret`, string 0 = high E, fret 0 = nut. Labels are physical, so horizontal flip uses identity flip indices.
   - Loss: smooth L1 with per-point weights: corners 1/4, 46 edge points 1/46, 76 internal 1/76.
   - Labels: the public GuiFiR release is GuiFiR-Vid only (clips, tablature, detector neck boxes; 4 corners `lt/lb/rt/rb` only on val 543 / test 2,048 frames, in 640×360 coords). **GuiFiR-Synth (126-point labels) is not released.** So we click 4 corners per frame (`scripts/01b_annotate_corners.py`) on ~1 frame per GuiFiR-Vid train clip plus GAPS train-split frames, and generate the 126 points from the corners with the fret formula d_n ∝ 1 − 2^(−n/12) plus a homography. Crossed/concave corner sets are rejected (`corners_valid`).
   - Annotation clicks: by default the 3rd/4th clicks go on **fret 12** (double-dot marker, nearly always visible; on classical guitars the neck/body joint, and classical guitars often have only 19 frets). Fret 20 is extrapolated with the same fret formula + homography (`fretboard_grid.corners_from_clicks`), so it is exact up to click noise. Key `f` switches to clicking fret 20 directly. Records always store nut + fret-20 `corners`, plus the raw `clicks` and `clicked_fret` for traceability.
   - Annotation UI: OpenCV window (kept on top under X11 via `python-xlib`, `pip install -e .[annotate]`), or `--web [PORT]` (default 8765) for SSH: a local browser page at http://localhost:8765/ with the same keys; VS Code forwards the port.
   - Annotation tool extras: `--no-shuffle` keeps sorted order; key `p` pastes the previous accepted clicks (also after resuming), for consecutive clips of one static-camera video.
   - Splits: train = our annotations; val = GuiFiR val corners; test = GuiFiR test corners (held out). Report FCD (mean corner error, px at 640×360) as in the paper.
   - GuiFiR corner mapping **verified** (Nov 16): `lt/lb/rt/rb` are image sides and the player faces the camera, so the **nut is on the image right**. Our order (nut/high-E, nut/low-E, f20/high-E, f20/low-E) = `rb, rt, lb, lt`. Evidence: fretting hand in the right half in all 1,483 low-fret val/test frames, right side narrower, overlays align.
   - GuiFiR frames: annotation names `<clip>/<clip>_NNN.png` are 0-based native-fps frame indices. The official `extract_frames.py` names frames differently and extracts everything (~300k PNGs), so `01a` uses `guifir.extract_annotated_frames` instead (~11 GB). 63 annotated frames have an index past the clip end (train 2, val 4, test 57; listed in `frames/missing.txt`) and are skipped → val 539, test 1,991 usable frames.
   - GuiFiR videos are mostly tab-tutorial videos (tab/score overlays on screen, mostly acoustic, front-facing camera).
   - Still needed for GAPS/own video at inference: a fretboard box detector (GuiFiR boxes only cover their own clips).
2. **Homography** from 4+ points, e.g. both ends of the nut and both ends of the 12th fret.
3. **Temporal smoothing** (moving average or Kalman) to stop warp jitter; reuse the last good warp when the hand occludes the neck.
4. **Warp to a fixed strip** (~800×144 grayscale).
   - Planned refinement: warp so every fret has equal width (undo the 2^(-1/12) fret spacing). Each pixel column then maps to a fixed fret, the guitar analogue of the S2S/V2N slope prior.

### Stage 2: transcription model (V2N-style)

Reference architecture from the V2N paper, to be adapted:

- Feature extractor: S2S ResNet-18 (grayscale, trained from scratch), 5-frame aggregation (0.2 s at 25 fps), slope prior (learned 1D encoding of horizontal position → key), global average pooling → T×512, linear projection with dropout 0.5.
- Temporal backbone: LayerNorm + 3 Conformer ConvModule blocks (FFN → depthwise 1D conv, kernel 31 → FFN, no self-attention). Receptive field covers the full 1 s (25-frame) window.
- Heads: parallel BiLSTM (hidden 256) → Linear. V2N has onset, offset, key hold, velocity.
- Training labels: soft triangular kernels (±2 frames) for onset/offset; binary hold over [onset, offset). Loss on all frames in the window, not only the centre frame.
- Decoding: onset peaks start notes; a note ends at the offset peak or when hold drops, whichever comes first.

Guitar adaptations decided so far:

- Keep **onset + hold (frame)** heads. Offset is optional and carries caveats. **Drop velocity.**
- Replace the keyboard warp with the fretboard warp from Stage 1.
- Replace the 88-key slope prior with a fretboard position prior (design open, see Open questions).

## Datasets

| Dataset | Role | Labels | Notes |
|---|---|---|---|
| **GAPS** | Main training set and real-MIDI evaluation | Real note-level MIDI (score-aligned) | 300 classical guitar performances, ~14 h, >250k notes, 205 performers, YouTube videos in varied recording conditions. Fretboard visibility not yet audited. Source: HF `xavriley/GAPS` v1.1 (MIT, audio included; v1 Zenodo zip has case-clashing file names, don't use). Splits from `gaps_metadata_with_splits.csv`: 270 train / 30 test; 101 rows without split are ignored. Videos via `yt_id` (YouTube). **Train split downloaded (Nov 25):** audio + MIDI 270/270 (14.85 h), videos **267/270** (14.71 h; missing `276_jw1wc` sign-in wall, `381_9V1wc` / `395_w41wc` removed from YouTube, list in `videos/failed_train.tsv`); 4 videos shorter than the audio by 1.7–18.6 s (`005_dSswc`, `055_cV1wc`, `227_x41wc`, `289_fY1wc`: check sync before use), 12 videos only 320×240. Some train wavs are WAVE_FORMAT_EXTENSIBLE (`gaps.wav_duration` falls back to ffprobe). **Sync:** GAPS audio/MIDI is earlier than the video by a constant per-piece lag (0–49 ms = codec priming delays); always measure it (`baselines.clips.measure_sync`) and shift MIDI when pairing with video. Many videos are multi-camera edits (cuts inside a piece). |
| **GuiFiR-Vid** | FPE corner labels (val/test); baseline evaluation set (CP7, test split, pseudo-GT) | Sparse tablature (~20 picking frames per 10 s clip: fret per string + capo; **manual on val/test** and 100 train clips, TabCNN pseudo-GT on the rest of train; `frame_tab[0]` = high E, 20 = not played, pitch = open + capo + fret), GT fretboard corners on val/test frames | 1,030 clips (900/30/100), mostly steel-string acoustic pop/anime covers from tab-tutorial YouTube videos, 640×360 @ 29.97 mostly, audio inside the mp4. Local: `data/raw/guifir/guifir_vid/`. |
| **magcil/guitar_style_dataset** | Baseline evaluation set (CP8, pseudo-GT); extra training data later | None time-aligned. Pseudo-MIDI from audio transcription of the wav files, spot-checked; MuseScore scores of the exercises (not aligned) | Electric guitar, 3 guitars × 3 amps, 9 playing techniques (alternate/sweep picking, bend, hammer-on, pull-off, legato, slide, tapping, vibrato), mp4 + separate wav. 549 recordings by **one guitarist** (smartphone, 1080p30, close front view; wav extracted from the mp4s; licence CC BY-NC-ND 4.0). Source: local `~/Downloads/guitar_style_dataset-v1.0.0.zip` (repo commit `eb27d7b`), extracted unchanged to `data/raw/magcil/` (see `SOURCE.txt`). Clips used in the baseline study are evaluation-only (exclude these 27 recordings if magcil becomes training data). |
| **Own recordings** | Multi-view data plus a carefully labelled test set | Audio-transcribed MIDI, manually checked for the test portion | Multi-view cameras, clean DI/mic audio, sync by clap or timecode. |

## Papers (PDFs in the claude.ai Project)

| Short name | Full reference | File | Used for |
|---|---|---|---|
| **V2N** | Kim, Sohn, Nam, Lerch. "Multi-Task Multi-Frame Visual Piano Transcription." ISMIR 2026. arXiv:2608.03419. Code and checkpoints: https://github.com/yonghyunk1m/V2N | `V2N.pdf` | Main architecture reference: multi-task heads, 1 s context, Conformer backbone, multi-frame loss, decoding. |
| **S2S** | Koepke, Wiles, Moses, Zisserman. "Sight to Sound: An End-to-End Approach for Visual Piano Transcription." ICASSP 2020. | `Sight_to_Sound_An_End-to-End_Approach_for_Visual_Piano_Transcription.pdf` | Feature extractor (ResNet-18, 5-frame aggregation, slope module); precedent for training on pseudo ground truth from audio transcription. |
| **PianoVAM** | Kim, Park, Bae, Kwon, Kim, Lerch, Nam. "PianoVAM: A Multimodal Piano Performance Dataset." ISMIR 2025. arXiv:2509.08800. | `PianoVAM.pdf` | Piano video/audio/MIDI dataset V2N is evaluated on; related work and reference for multimodal data collection and sync. |
| **GAPS** | Riley, Guo, Edwards, Dixon. "GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model." ISMIR 2024. | `GAPS_DATASET_PAPER.pdf` | Primary dataset (real MIDI + performance video). |
| **magcil** | Mitsou, Petrogianni, Vakalaki, Nikou, Psallidas, Giannakopoulos. "A multimodal dataset for electric guitar playing technique recognition." Data in Brief 52 (2024) 109842. doi 10.1016/j.dib.2023.109842 | `magcil_guitar_style_dataset.pdf` | Third baseline dataset (electric guitar, 9 techniques, video + wav, no note labels). |
| **Foley Music** | Gan, Huang, Chen, Tenenbaum, Torralba. "Foley Music: Learning to Generate Music from Videos." ECCV 2020, LNCS 12356, pp. 758–775. | `Foley-Music.pdf` | Optional CP14 baseline and closest related work: video → body/hand keypoints → **MIDI** (Graph-Transformer), then synthesised; trained on URMP, AtinPiano, MUSIC (incl. guitar). |
| **HunyuanVideo-Foley** | Shan, Li, Cui, Yang, Wang, Yang, Zhou, Zhong. "HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation." arXiv:2508.16930, 2025. | `HunyuanVideo-Foley.pdf` | Optional CP14 V2A baseline (text+video → audio, multimodal diffusion transformer, 100k h training data). |
| **ThinkSound** | Liu, Luo, Wang, Wang, Chen, Zhao, Xue. "ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing." NeurIPS 2025. arXiv:2506.21448. | `ThinkSound.pdf` | Optional CP14 V2A baseline (MLLM chain-of-thought guiding an audio foundation model). |
| **GuiFiR** | An, Chung, Park, Lee. "GuiFiR: Weakly-Supervised Learning of Hand Pose Estimation for Guitar Finger Reading." IEEE Access, vol. 14, 2026. DOI 10.1109/ACCESS.2026.3713562. | `GuiFiR_Weakly-Supervised_Learning_of_Hand_Pose_Estimation_for_Guitar_Finger_Reading.pdf` | Basis for the fretboard warp step: joint hand and fretboard pose estimation, trained on synthetic composites (GuiFiR-Synth) plus real videos with audio-derived tablature (GuiFiR-Vid). Datasets are public. |

Secondary references for fretboard tracking:

- Synthetic keypoint detection: https://ieeexplore.ieee.org/document/10390631/
- Modified RANSAC neck tracking: https://library.imaging.org/ei/articles/30/18/art00005
- Fret-geometry homography (ISMIR 2010): https://ismir2010.ismir.net/proceedings/ismir2010-29.pdf
- GuiFiR page: https://ailab.kookmin.ac.kr/publications/2026-07-15-guifir-weakly-supervised-learning-of-han/

## Key decisions and why

| Decision | Date | Why |
|---|---|---|
| Target is MIDI, not audio | Nov 13 | Fits the deadline; measurable with standard note metrics; direct analogue of visual piano transcription. |
| Adapt V2N / S2S instead of designing a new model | Nov 13 | Proven on a closely related task; clear baseline and ablation structure. |
| Data = GAPS + magcil + own multi-view | Nov 13 | GAPS has real MIDI and video; magcil adds video volume; own data gives controlled views and a clean test set. |
| Rectify the fretboard before the model | Nov 11 | V2N depends on a fixed-geometry strip. V2N's cross-dataset transfer collapses to near zero when geometry differs, so geometry normalisation matters even more for unconstrained guitar video. |
| Equal-fret-width warp | Nov 11 | Makes pixel column correspond to fret, giving the model a positional prior like the slope prior. |
| GuiFiR for the warp step | Nov 13 | Provides a dense fretboard grid; dataset is public. |
| Train the fretboard estimator ourselves | Nov 16 | No released FPE weights or GuiFiR-Synth; only GuiFiR-Vid is public. |
| Reconstruct only the GuiFiR FPE, not the full model | Nov 16 | The warp needs only fretboard points; the hand estimator / TabCNN / joint training would risk the timeline. |
| FPE labels from manually clicked 4 corners → 126-point grid | Nov 16 | GuiFiR-Synth is not public; the paper itself derives grids from 4 corners. Annotating GAPS frames also serves the GAPS visibility audit. |
| Click nut + fret 12 instead of fret 20 when annotating corners | Nov 19 | Fret 20 is often hidden by the hand/body or absent (19-fret classical guitars); fret 12 is marked and visible, and fret 20 follows exactly from the fret formula. |
| Drop the velocity head | Nov 11 | GAPS velocities are not real. |
| Python 3.10 + PyTorch 2.1.2 | Nov 16 | All reference code (V2N, GuiFiR, GAPS) and needed libraries (OpenCV, pretty_midi, mir_eval) are Python. 3.10 / torch 2.1 because mmcv 2.1.0 has prebuilt wheels only for those (planned 3.11 changed). |
| Evaluate only on real MIDI | Nov 13 | Pseudo-labels (magcil) would make test numbers meaningless. Relaxed Nov 24 for the baseline study only (see below). |
| Baseline study of existing models before more FPE work; GuiFiR resumes afterwards, including hand pose | Nov 23 | Supervisor: get a bird's-eye view of how bad existing video→audio/MIDI models are on guitar first; this is also the thesis motivation. |
| Score V2A models via audio transcription, with the real-audio transcription as ceiling | Nov 23 | V2A models output audio, not MIDI; the ceiling separates V2A error from transcription error. |
| V2N zero-shot uses strips from manually clicked corners, tracked through each clip | Nov 23 | Exact, fair crop so V2N's failure is due to piano→guitar, not cropping; FPE not trained yet. Tracking because GAPS has zoom/pan, players moving the guitar and cuts: one fixed crop per clip would let the fretboard slide out of the strip. First version of Stage 1 steps 2–4. |
| Corner clicks on GAPS **test** clips are evaluation-only, never FPE training data | Nov 23 | Training the FPE on test frames would leak into the GAPS test evaluation. Stored as `gaps_test_baseline_EVAL_ONLY.jsonl`. Only CP8 clicks (GAPS train split) may become FPE training data. |
| Baseline evaluation masked to frames with a valid fretboard view, for all systems | Nov 23 | After cuts/dissolves V2N sees nothing; charging it for those notes would measure our cropping, not V2N. Same mask for every system keeps the comparison fair. |
| Baseline study extended to GuiFiR-Vid and magcil before the results table | Nov 24 | One dataset (classical concert video) is too narrow for "how bad are existing models"; GuiFiR (steel-string, tutorials) and magcil (electric, techniques) add other guitars and recording setups. |
| Pseudo-GT datasets in the same baseline table, with a label-source column | Nov 24 | Luckey's choice. GuiFiR/magcil have no real MIDI; reference = GAPS model on their real audio. Trained-model main results (CP13+) stay real-MIDI only; pseudo-GT sets only as secondary cross-dataset rows. |
| Full-string strips (nut → bridge) as CP12, before the V2N fine-tune | Nov 24 | The current strips show the fretting hand only; the picking hand (when a note sounds) is outside. Bridge follows from the tracked corners (fret 12 = half the scale), so no new clicks; checking the extrapolated part on the test clips first is cheap and needed before the training strips depend on it. Fine-tune moves to CP13. |
| GuiFiR tab = reliability check of the reference + extra "pitch recall at tab frames" column | Nov 24 | Only independent label on GuiFiR; sparse and without onsets/offsets, so recall only, not a replacement for the reference. |
| Checkpoints: one per dataset for its preprocessing, then one per all-dataset task | Nov 24 | Readability: a CP row says what was done to which data. Old CP7/8/9 → CP11/12/13. |
| V2N on guitar: full fine-tune vs scratch; LoRA only as optional ablation | Nov 23 | V2N is small (ResNet-18 + conv blocks + BiLSTM, no attention) and fits 16 GB; LoRA saves little and suits conv/LSTM layers poorly; heads must be replaced anyway. |

## Evaluation

- Note-level F1 with `mir_eval` (onset; onset+offset where offsets are used), plus frame-level F1.
- V2N reports onset tolerances of 50 ms and 100 ms; follow the same convention. At 25 fps the time grid is 40 ms.
- Test sets: GAPS (real MIDI) and the carefully labelled own test set. **Trained models: main numbers only on real MIDI** (GAPS test, own test set); GuiFiR-Vid / magcil may appear only as secondary, labelled cross-dataset rows against pseudo-GT (decided Nov 24 for CP13). Exception (Nov 24): the baseline study (CP7–CP11) also reports GuiFiR-Vid and magcil against a pseudo-GT reference (GAPS model on the real audio), always with the label source shown.

## Timeline

| Dates | Phase | Done when |
|---|---|---|
| Nov 13–22 | Setup and audit | GAPS fretboard-visibility audit; recording protocol; supervisor sign-off; submission rules checked |
| Nov 16–30 | Record own data | Multi-view sessions recorded and synced, clean DI/mic audio |
| Nov 23–Dec 6 | Data pipeline | Landmarks → homography → smoothed warped strips; MIDI labels for all sets; splits |
| Dec 7–20 | Baseline | V2N-style model on GAPS single view, onset + frame heads, note F1 |
| Dec 21–Jan 6 | Main experiments | Add magcil pseudo-labels, multi-view / picking-hand stream, ablations. **Freeze Jan 6** |
| Dec (parallel) | Write early | Related work and Method |
| Jan 7–22 | Write | Results, failure analysis, discussion, abstract |
| Jan 23–Feb 1 | Review and buffer | Supervisor feedback, proofreading, formatting, code repo |

### Fallback rules

- Own data not done by Nov 30 → single view only. Multi-view is a stretch goal.
- Fretboard warp not stable by Dec 6 → fixed neck crop.
- No new experiments after Jan 6.

## Open questions

1. **GAPS audit:** how many hours show the fretboard clearly enough to use?
2. **Fretboard estimator:** decided: GuiFiR FPE trained on our own corner annotations (see Stage 1). Open: how accurate it gets on real frames (fake-data check reached ~26 px median error on 640×360 after 100 epochs, still improving); which box detector to use on GAPS; whether the full 126 points or just 4 corners give the more stable homography.
3. **Output representation:** pitch only, or string × fret (tablature)? This determines head size and the form of the position prior. Alternate tunings in GAPS need handling either way.
4. **Offset head:** keep or drop? Guitar notes decay and are damped, so offsets are less visually defined than piano key releases.
5. **Picking-hand stream:** add a second crop of the picking hand as input? Onsets are produced by the picking hand; the fretting hand mostly gives pitch. Current strips (nut → fret 20) mostly miss the picking hand (and open strings / repeated notes are invisible from the fretting hand). First step (Nov 24): CP12 full-string strip nut → bridge (one input, no model change) and CP13 run D; a separate picking-hand stream (second encoder) or hand pose stays for the main experiments.
6. **Multi-view fusion:** how to combine views, if own multi-view data is ready in time.
7. **Pseudo-label quality:** which audio transcription model for magcil and own data, and how much spot-checking is enough?
8. **Frame rate:** resample all video to 25 fps as V2N does, or keep higher rates for fast passages?
9. **Supervisor sign-off and submission rules:** still pending.

## Project structure

```
configs/{data,preprocessing,model}/   configs, copied into each run folder (YAML; MMPose configs are Python)
  preprocessing/fpe_res50_regression.py   FPE model/data/training config
  preprocessing/fretboard_126_metainfo.py 126 keypoint names, loss weights, identity flip
data/          (gitignored) raw/{gaps,magcil,guifir,own} → interim/{landmarks,homographies} → processed/strips, labels/, splits/
               FPE: labels/fretboard_corners/*.jsonl (clicked corners), processed/fretboard_coco/{train,val,test}.json
src/guitar2midi/   importable package: data, preprocessing, model, training, decoding, evaluation, utils
  data/guifir.py                    GuiFiR-Vid reader (frames, neck boxes, GT corners, lt/lb/rt/rb mapping, extract_annotated_frames)
  data/magcil.py                    magcil guitar_style_dataset reader (file-name metadata: technique, guitar, amp, exercise)
  data/gaps.py                      GAPS v1.1 reader/downloader: metadata+splits, audio/MIDI (HF), videos (yt-dlp) → data/raw/gaps/{audio,midi,videos}/<id>.*
  baselines/v2a_mmaudio.py          MMAudio wrapper (env mmaudio only): model loaded once, weights in ~/.cache/mmaudio, mux for listening
  baselines/v2n.py                  V2N wrapper (env v2n only): V2N's own model/stitching/decoder from ~/external/V2N (commit a6408c4, override with $V2N_REPO) on our .npy strips; checkpoint in ~/.cache/v2n; GT-vs-V2N roll plot; transform_strips (flip / piano-like layout / brightness-norm variants, CP10)
  baselines/audio_plots.py          real vs generated spectrograms with GT notes overlaid
  baselines/transcribe.py           audio→MIDI with the GAPS benchmark model (env amt only)
  evaluation/note_metrics.py        onset F1 @50/100 ms, onset+offset F1, frame F1 (25 fps), timing F1 (pitch ignored), chroma similarity, onset correlation (mir_eval)
  baselines/clips_guifir.py         GuiFiR-Vid eval clips: window choice, audio cut, tab readings (frame_tab[0] = high E, + capo), tab_recall
  baselines/clips.py                eval clips for the baseline study: window selection, cutting, sync measurement + correction, contact sheet, piano-roll plot
  preprocessing/fretboard_grid.py   126-point grid from 4 corners, corners_valid, draw_grid, extend_corners (nut → bridge, BRIDGE_X = 1.46)
  preprocessing/fpe_inference.py    FretboardPoseEstimator.predict(image, boxes) -> (N, 126, 2)
  preprocessing/video.py            read_frames, detect_cuts (colour-histogram drop or pixel-difference jump), shots
  preprocessing/fretboard_tracking.py  track_corners (masked LK flow + RANSAC homography, hold last good, smoothing), warp_strip (800×144, nut left), tracking_block/save_sheet (visual check)
scripts/       thin numbered entry points
  done:    setup_mmpose.sh, 01a_prepare_guifir.sh, 01b_annotate_corners.py, 01c_build_fretboard_coco.py,
           01d_train_fpe.py, 01e_eval_fpe.py
  planned: 00_audit_gaps, 02_warp_strips, 03_make_labels, 04_make_splits, 10_train, 20_evaluate
  baseline study: done 30a_download_gaps (--split test|train, --no-audio, --no-videos), 30b_make_eval_clips (→ data/interim/baseline_clips/), 30c_run_mmaudio (env mmaudio), 30c_plot_spectrograms, 30d_transcribe (env amt), 30e_evaluate_baselines (→ outputs/runs/);
           30f_prepare_strip_frames (cuts + frames to click), 30f_make_v2n_strips (tracking + strips);
           30g_run_v2n (env v2n, → transcriptions/v2n_zeroshot), 30g_check_v2n_piano (positive control on one PianoVAM video in data/raw/pianovam);
           31a_make_guifir_clips + 31a_check_guifir_tab (CP7, → data/interim/baseline_clips_guifir/; 30d takes --clips-dir);
           31b_make_magcil_clips + 31b_spot_check_magcil (CP8, → data/interim/baseline_clips_magcil/); 30f_* take --clips-dir, 30f_prepare_strip_frames --no-cuts;
           CP9/CP10: 30c_run_mmaudio, 30c_plot_spectrograms, 30e (--clips <dir>; reference = midi/ or transcriptions/real_audio, per-clip length_s), 30g_run_v2n (--clips-dir, --flip/--layout/--norm variants), 30g_v2n_recheck, 30g_strip_stats;
           CP11: 30i_results_table (→ thesis/tables/baseline_results.tex; main.tex loads multirow);
           CP12: 30j_make_full_strips (→ <clips-dir>/strips_full/, from strips/<clip>_corners.npy);
           planned 30h_finetune_v2n (CP13)
notebooks/     exploration only, numbered
outputs/       (gitignored) runs/<YYYY-MM-DD>_<name>/ (mmengine work_dir: merged config, checkpoints, logs)
tests/         unit tests (test_fretboard_grid.py: grid geometry, weights, metainfo matches grid module; test_note_metrics.py: metric sanity)
envs/          extra conda envs for external models (amt.yml, mmaudio.yml, v2n.yml)
docs/papers/   reference PDFs; docs/notes/ protocol, audit results, meeting notes
thesis/        LaTeX source (main.tex, chapters/, figures/, tables/; build/ gitignored)
```

Environment (Miniforge installed at `~/miniforge3`, env `guitar2midi`, Python 3.10):
`conda env create -f environment.yml && conda activate guitar2midi && bash scripts/setup_mmpose.sh`.
Extra envs for the baseline study (each: `conda env create -f envs/<name>.yml`, then `pip install -e . --no-deps`): `amt` (GAPS audio-to-MIDI model), `mmaudio` (MMAudio V2A) and `v2n` (V2N; also needs the repo clone + checkpoint download listed in `envs/v2n.yml`). Separate because all need a recent torch (2.15), which breaks mmcv in `guitar2midi`. Scripts state which env they need.
LaTeX preview (no sudo, no full TeX Live): env `latex` (`conda create -n latex -c conda-forge tectonic`); `thesis/build/preview/table_preview.tex` inputs a generated table → `tectonic table_preview.tex && pdftoppm -png -r 150 -singlefile table_preview.pdf table_preview` (build/ is gitignored).
For the GuiFiR download also `pip install -e .[data]` (gdown, yt-dlp); for the corner annotation window `pip install -e .[annotate]` (python-xlib). Run tests with `python -m pytest tests/`.
Pinned because MMPose is fragile: torch 2.1.2 + cu121, mmcv 2.1.0, mmengine 0.10.7, mmdet 3.3.0 (imported by mmpose heads), mmpose 1.3.2 with `--no-deps` (chumpy fails to build), numpy<2, setuptools<81 (mmengine needs `pkg_resources`).

FPE workflow: `01a_prepare_guifir.sh` (download) → `01b_annotate_corners.py` → `01c_build_fretboard_coco.py --manual ... --visualize 20` → `01d_train_fpe.py --run-name ...` → `01e_eval_fpe.py --run outputs/runs/<run>`. Config: `configs/preprocessing/fpe_res50_regression.py` (paths relative to `data_root='data/'`).

## Working conventions for Claude Code

- Keep the two stages separate in code: `src/guitar2midi/preprocessing/` (landmarks, homography, smoothing, warp) and `src/guitar2midi/model/` (V2N-style network); training, decoding and evaluation have their own subpackages.
- Scripts only load a config and call package functions; no logic in scripts or notebooks that other steps depend on.
- Each pipeline stage reads from one `data/` subfolder and writes to the next, so stages can be rerun independently.
- Every training/eval run writes to its own `outputs/runs/<date>_<name>/` with a copy of its config, so thesis numbers are traceable.
- Thesis figures and tables are generated by scripts or notebooks into `thesis/figures/` and `thesis/tables/`, not edited by hand.
- Never commit videos, strips, audio or checkpoints.
- Cache warped strips to disk; never recompute the warp inside the training loop.
- Keep train/test splits by performer and by recording, to avoid leakage.
- Mark every label source explicitly (real MIDI vs pseudo-label) in dataset metadata.
- Prefer the simplest thing that meets the next timeline milestone; the fallback rules above take priority over ambition.