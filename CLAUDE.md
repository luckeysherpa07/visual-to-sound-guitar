# CLAUDE.md — Guitar Video → MIDI Thesis

Master's thesis (Computer Science, JMU Würzburg). Author: Ruluckey "Luckey" Sherpa. Supervisor: Zongwei Wu.
**Hard deadline: Feb 1, 2027. Experiment freeze: Jan 6, 2027.**
Status as of 2026-11-16: planning done, no code yet. Update this file when decisions change.

## Goal

Predict MIDI notes from **silent video** of guitar playing (visual music transcription). The approach adapts the visual piano transcription pipelines V2N and Sight-to-Sound (S2S) to guitar.

Scope is locked to MIDI prediction. Audio generation and performance feedback (earlier ideas) are out of scope.

## Input / output

- **Input:** guitar performance video, no audio at inference. Frames are preprocessed into a rectified, fixed-size, grayscale fretboard strip (target ~800×144, as in V2N), in 1 s clips at 25 fps.
- **Output:** note events (pitch, onset, and offset where reliable) written as MIDI.
- **Not predicted:** velocity. GAPS velocities are not real, so there is no trustworthy supervision.

## Pipeline

```
video ─► fretboard landmarks ─► homography ─► temporal smoothing ─► warped strip
      ─► V2N-style model ─► onset / frame (hold) [/ offset] heads ─► peak decoding ─► MIDI
```

### Stage 1: fretboard preprocessing (current focus)

1. **Landmark detection per frame.** Method: GuiFiR-style fretboard estimator (GuiFiR predicts a 126-landmark fretboard grid plus hand pose; its dataset is public).
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
| **GAPS** | Main training set and real-MIDI evaluation | Real note-level MIDI (score-aligned) | 300 classical guitar performances, ~14 h, >250k notes, 205 performers, YouTube videos in varied recording conditions. Fretboard visibility not yet audited. |
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
| Train the fretboard estimator ourselves, not rely on released weights | Nov 16 | **Tentative**, still being considered. |
| Drop the velocity head | Nov 11 | GAPS velocities are not real. |
| Python 3.11 + PyTorch | Nov 16 | All reference code (V2N, GuiFiR, GAPS) and needed libraries (OpenCV, pretty_midi, mir_eval) are Python. |
| Evaluate only on real MIDI | Nov 13 | Pseudo-labels (magcil) would make test numbers meaningless. |

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
2. **Fretboard estimator:** train from scratch on the GuiFiR dataset, use released weights if available, or a synthetic-render keypoint model? How many landmarks are actually needed for a stable homography?
3. **Output representation:** pitch only, or string × fret (tablature)? This determines head size and the form of the position prior. Alternate tunings in GAPS need handling either way.
4. **Offset head:** keep or drop? Guitar notes decay and are damped, so offsets are less visually defined than piano key releases.
5. **Picking-hand stream:** add a second crop of the picking hand as input? Onsets are produced by the picking hand; the fretting hand mostly gives pitch.
6. **Multi-view fusion:** how to combine views, if own multi-view data is ready in time.
7. **Pseudo-label quality:** which audio transcription model for magcil and own data, and how much spot-checking is enough?
8. **Frame rate:** resample all video to 25 fps as V2N does, or keep higher rates for fast passages?
9. **Supervisor sign-off and submission rules:** still pending.

## Project structure

```
configs/{data,preprocessing,model}/   YAML configs, copied into each run folder
data/          (gitignored) raw/{gaps,magcil,own} → interim/{landmarks,homographies} → processed/strips, labels/, splits/
src/guitar2midi/   importable package: data, preprocessing, model, training, decoding, evaluation, utils
scripts/       thin numbered entry points (00_audit_gaps, 01_extract_landmarks, 02_warp_strips, 03_make_labels, 04_make_splits, 10_train, 20_evaluate)
notebooks/     exploration only, numbered
outputs/       (gitignored) runs/<YYYY-MM-DD>_<name>/{config.yaml,checkpoints,metrics,logs}
tests/         unit tests
docs/papers/   reference PDFs; docs/notes/ protocol, audit results, meeting notes
thesis/        LaTeX source (main.tex, chapters/, figures/, tables/; build/ gitignored)
```

Environment: `conda env create -f environment.yml` (installs the package with `pip install -e .[dev]`).

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