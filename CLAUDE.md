# CLAUDE.md — Guitar Video → MIDI Thesis

Master's thesis (Computer Science, JMU Würzburg). Author: Ruluckey "Luckey" Sherpa. Supervisor: Zongwei Wu.
**Hard deadline: Feb 1, 2027. Experiment freeze: Jan 6, 2027.**
Status as of 2026-11-23: after the supervisor meeting on Nov 23, the FPE work (Stage 1) is **paused** and the current phase is the **baseline study** (see below): measure how badly existing off-the-shelf models do on our guitar data before building our own. After that, GuiFiR work resumes, now including the hand pose estimator. FPE state when paused: code written and verified end to end on fake fretboards, GuiFiR-Vid prepared (val/test COCO files), corner annotation tool finished, 3 train frames clicked, no real training run. Update this file when decisions change.

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
- **Data:** GAPS test split, ~20 pieces × 3 clips × 8 s. magcil only as pseudo comparison; own data later with the same scripts.
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
| 6 | V2N zero-shot on the strips (public checkpoints) | predicted vs GT piano roll | todo |
| 7 | Results table: ceiling / MMAudio ± prompt / V2N zero-shot / shuffled video / silence | table + summary for supervisor | MMAudio part done (Nov 23): note F1 **3.3 %** vs ceiling 93.5 % (chance 2.1 %); timing F1 35.9 vs chance ~27 (significant); chroma = chance; prompt no effect. V2N rows after CP6 |
| 8 | V2N fine-tuning pilot: guitar heads, GAPS train strips (1 click set per video); full fine-tune vs scratch (LoRA optional) | curves + table rows | todo |
| 9 | Optional: more V2A models, magcil (pseudo), own data | extra table rows | todo |

Targets: CP7 by ~Nov 28, CP8 by ~Dec 4. If CP8 has no results after ~4 days, move it to the main experiments phase.

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
| **GAPS** | Main training set and real-MIDI evaluation | Real note-level MIDI (score-aligned) | 300 classical guitar performances, ~14 h, >250k notes, 205 performers, YouTube videos in varied recording conditions. Fretboard visibility not yet audited. Source: HF `xavriley/GAPS` v1.1 (MIT, audio included; v1 Zenodo zip has case-clashing file names, don't use). Splits from `gaps_metadata_with_splits.csv`: 270 train / 30 test; 101 rows without split are ignored. Videos via `yt_id` (YouTube). **Sync:** GAPS audio/MIDI is earlier than the video by a constant per-piece lag (0–49 ms = codec priming delays); always measure it (`baselines.clips.measure_sync`) and shift MIDI when pairing with video. Many videos are multi-camera edits (cuts inside a piece). |
| **magcil/guitar_style_dataset** | Extra training data | None. Pseudo-MIDI from audio transcription of the wav files, spot-checked | Chosen because it has the needed video. Training only. |
| **Own recordings** | Multi-view data plus a carefully labelled test set | Audio-transcribed MIDI, manually checked for the test portion | Multi-view cameras, clean DI/mic audio, sync by clap or timecode. |

## Papers (PDFs in the claude.ai Project)

| Short name | Full reference | File | Used for |
|---|---|---|---|
| **V2N** | Kim, Sohn, Nam, Lerch. "Multi-Task Multi-Frame Visual Piano Transcription." ISMIR 2026. arXiv:2608.03419. Code and checkpoints: https://github.com/yonghyunk1m/V2N | `V2N.pdf` | Main architecture reference: multi-task heads, 1 s context, Conformer backbone, multi-frame loss, decoding. |
| **S2S** | Koepke, Wiles, Moses, Zisserman. "Sight to Sound: An End-to-End Approach for Visual Piano Transcription." ICASSP 2020. | `Sight_to_Sound_An_End-to-End_Approach_for_Visual_Piano_Transcription.pdf` | Feature extractor (ResNet-18, 5-frame aggregation, slope module); precedent for training on pseudo ground truth from audio transcription. |
| **PianoVAM** | Kim, Park, Bae, Kwon, Kim, Lerch, Nam. "PianoVAM: A Multimodal Piano Performance Dataset." ISMIR 2025. arXiv:2509.08800. | `PianoVAM.pdf` | Piano video/audio/MIDI dataset V2N is evaluated on; related work and reference for multimodal data collection and sync. |
| **GAPS** | Riley, Guo, Edwards, Dixon. "GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model." ISMIR 2024. | `GAPS_DATASET_PAPER.pdf` | Primary dataset (real MIDI + performance video). |
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
| Evaluate only on real MIDI | Nov 13 | Pseudo-labels (magcil) would make test numbers meaningless. |
| Baseline study of existing models before more FPE work; GuiFiR resumes afterwards, including hand pose | Nov 23 | Supervisor: get a bird's-eye view of how bad existing video→audio/MIDI models are on guitar first; this is also the thesis motivation. |
| Score V2A models via audio transcription, with the real-audio transcription as ceiling | Nov 23 | V2A models output audio, not MIDI; the ceiling separates V2A error from transcription error. |
| V2N zero-shot uses strips from manually clicked corners, tracked through each clip | Nov 23 | Exact, fair crop so V2N's failure is due to piano→guitar, not cropping; FPE not trained yet. Tracking because GAPS has zoom/pan, players moving the guitar and cuts: one fixed crop per clip would let the fretboard slide out of the strip. First version of Stage 1 steps 2–4. |
| Corner clicks on GAPS **test** clips are evaluation-only, never FPE training data | Nov 23 | Training the FPE on test frames would leak into the GAPS test evaluation. Stored as `gaps_test_baseline_EVAL_ONLY.jsonl`. Only CP8 clicks (GAPS train split) may become FPE training data. |
| Baseline evaluation masked to frames with a valid fretboard view, for all systems | Nov 23 | After cuts/dissolves V2N sees nothing; charging it for those notes would measure our cropping, not V2N. Same mask for every system keeps the comparison fair. |
| V2N on guitar: full fine-tune vs scratch; LoRA only as optional ablation | Nov 23 | V2N is small (ResNet-18 + conv blocks + BiLSTM, no attention) and fits 16 GB; LoRA saves little and suits conv/LSTM layers poorly; heads must be replaced anyway. |

## Evaluation

- Note-level F1 with `mir_eval` (onset; onset+offset where offsets are used), plus frame-level F1.
- V2N reports onset tolerances of 50 ms and 100 ms; follow the same convention. At 25 fps the time grid is 40 ms.
- Test sets: GAPS (real MIDI) and the carefully labelled own test set. **Never report results on magcil pseudo-labels.**

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
5. **Picking-hand stream:** add a second crop of the picking hand as input? Onsets are produced by the picking hand; the fretting hand mostly gives pitch.
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
  data/gaps.py                      GAPS v1.1 reader/downloader: metadata+splits, audio/MIDI (HF), videos (yt-dlp) → data/raw/gaps/{audio,midi,videos}/<id>.*
  baselines/v2a_mmaudio.py          MMAudio wrapper (env mmaudio only): model loaded once, weights in ~/.cache/mmaudio, mux for listening
  baselines/audio_plots.py          real vs generated spectrograms with GT notes overlaid
  baselines/transcribe.py           audio→MIDI with the GAPS benchmark model (env amt only)
  evaluation/note_metrics.py        onset F1 @50/100 ms, onset+offset F1, frame F1 (25 fps), timing F1 (pitch ignored), chroma similarity, onset correlation (mir_eval)
  baselines/clips.py                eval clips for the baseline study: window selection, cutting, sync measurement + correction, contact sheet, piano-roll plot
  preprocessing/fretboard_grid.py   126-point grid from 4 corners, corners_valid, draw_grid
  preprocessing/fpe_inference.py    FretboardPoseEstimator.predict(image, boxes) -> (N, 126, 2)
  preprocessing/video.py            read_frames, detect_cuts (colour-histogram drop or pixel-difference jump), shots
  preprocessing/fretboard_tracking.py  track_corners (masked LK flow + RANSAC homography, hold last good, smoothing), warp_strip (800×144, nut left)
scripts/       thin numbered entry points
  done:    setup_mmpose.sh, 01a_prepare_guifir.sh, 01b_annotate_corners.py, 01c_build_fretboard_coco.py,
           01d_train_fpe.py, 01e_eval_fpe.py
  planned: 00_audit_gaps, 02_warp_strips, 03_make_labels, 04_make_splits, 10_train, 20_evaluate
  baseline study: done 30a_download_gaps (--split test|train, --no-audio, --no-videos), 30b_make_eval_clips (→ data/interim/baseline_clips/), 30c_run_mmaudio (env mmaudio), 30c_plot_spectrograms, 30d_transcribe (env amt), 30e_evaluate_baselines (→ outputs/runs/);
           30f_prepare_strip_frames (cuts + frames to click), 30f_make_v2n_strips (tracking + strips);
           planned 30g_run_v2n.sh, 30h_finetune_v2n
notebooks/     exploration only, numbered
outputs/       (gitignored) runs/<YYYY-MM-DD>_<name>/ (mmengine work_dir: merged config, checkpoints, logs)
tests/         unit tests (test_fretboard_grid.py: grid geometry, weights, metainfo matches grid module; test_note_metrics.py: metric sanity)
envs/          extra conda envs for external models (amt.yml, mmaudio.yml)
docs/papers/   reference PDFs; docs/notes/ protocol, audit results, meeting notes
thesis/        LaTeX source (main.tex, chapters/, figures/, tables/; build/ gitignored)
```

Environment (Miniforge installed at `~/miniforge3`, env `guitar2midi`, Python 3.10):
`conda env create -f environment.yml && conda activate guitar2midi && bash scripts/setup_mmpose.sh`.
Extra envs for the baseline study (each: `conda env create -f envs/<name>.yml`, then `pip install -e . --no-deps`): `amt` (GAPS audio-to-MIDI model) and `mmaudio` (MMAudio V2A). Separate because both need a recent torch (2.15), which breaks mmcv in `guitar2midi`. Scripts state which env they need.
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