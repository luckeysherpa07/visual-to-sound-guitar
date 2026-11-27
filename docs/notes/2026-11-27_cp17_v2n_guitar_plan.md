# CP17 — V2N trained on guitar (approved plan)

Status: **approved 2026-11-27** (Luckey). Supersedes the draft `2026-11-24_v2n_guitar_adaptation_plan.md`, which
stays as background (architecture analysis: which V2N parts are piano-specific). Run folders:
`outputs/runs/<date>_cp17_v2n_<A|B|C|D>/`. Scripts are numbered `35x` (CP16 used `34x`).

## Question

CP6/CP10/CP12: the released V2N (piano) outputs silence on guitar strips. Can the V2N *architecture* learn
guitar notes from silent fretboard video when it is trained on GAPS (real MIDI), and does piano pretraining,
generic ImageNet pretraining or seeing the picking hand help?

## Decisions (Nov 27)

| Decision | Why |
|---|---|
| Model = V2N's own `Transcriber_OnsetFrame` (Conformer ×3 + BiLSTM onset/frame heads), 88 outputs, no model-code change | Velocity has no real labels in GAPS, offsets are weak on guitar (CLAUDE.md); guitar MIDI 40–83 lies inside the 88 keys, so the piano heads can be warm-started without surgery |
| Piano init (A, D): our wrapper loads `v2n_pianovam.safetensors` with `strict=False`, drops `offset_head.*` / `velocity_head.*`, asserts every other key loads | V2N's `train.py` only resumes runs, it has no "init from checkpoint" option; layer names (`proj`, `backbone`, `onset_head`, `frame_head`) match the piano model |
| Same performer split as CP16 (`data/processed/mmaudio_lora/split.csv`: 239 train / 28 val pieces) | CP16 (LoRA) and CP17 train and validate on the same pieces → fair comparison; split is by performer |
| GAPS test results also reported for seen vs unseen performers (12 of 27 test performers are in train) | As in CP16; no test performer is dropped from training (decided in CP16a) |
| Clicking: longest shots per video until ≥ 70 % of its length is covered, at most 5 shots, shots ≥ 2 s | Luckey (Nov 27). Many GAPS videos are multi-camera edits; tracking only works inside a clicked shot, so one click per video would lose most of an edited video. Cap keeps the clicking at ~2–3 h |
| GAPS train clicks are stored as `data/labels/fretboard_corners/gaps_train.jsonl` and may become FPE training data | Train split only, no test leakage (GAPS test clicks stay EVAL_ONLY) |
| ~20k steps per run (V2N: 100k); V2N's optimiser settings (AdamW lr 5e-4, cosine, warmup 5 %, bf16) | ~10 h per run on the RTX 4080 → 4 runs ≈ 40 h, fits ~Dec 12 |
| Training data GAPS only; GuiFiR-Vid and magcil stay unseen cross-dataset test sets | Real MIDI only for training; pseudo-GT sets only as labelled secondary rows |

## Runs (order A → D → B → C; C dropped first, then B, if time runs out)

| run | backbone | init | strips | question |
|---|---|---|---|---|
| A (main) | S2S ResNet-18 | V2N piano checkpoint, full fine-tune | nut → fret 20 | does piano visual knowledge transfer? |
| D | S2S ResNet-18 | as A | nut → bridge (CP12) | does seeing the picking hand help (onsets, open strings, repeated notes)? |
| B | ConvNeXt-T `fb_in22k` | ImageNet-22k | nut → fret 20 | generic image pretraining vs piano pretraining |
| C | S2S ResNet-18 | scratch | nut → fret 20 | lower bound: guitar data alone |

## Steps

- **17a — shots + click frames.** `35a_prepare_train_click_frames.py`: streaming cut detection on all 267
  train+val videos (ffmpeg thumbnails at 25 fps, same detector as CP5), shot choice as above, one full-resolution
  frame per chosen shot → `data/interim/v2n_train/click_frames/<piece>__s<k>/f<frame>.png` + `shots.json`.
  Luckey clicks with `01b_annotate_corners.py` (`p` pastes for repeated camera angles; `s` skips shots without a
  usable fretboard view). Output: shot statistics (cuts per video, coverage), click count.
- **17b — strips + labels.** (Built Nov 27: the CP5 tracker drifts over whole shots, so the keyframe tracker `track_corners_keyframe` is used; uncovered stretches ≥ 10 s get a second click round `35a --round2`; see log.) Track each clicked shot (whole shot, not 8 s), write both strip
  versions (CP5 fretboard, CP12 full string) from the same tracking, valid runs ≥ 2 s → one V2N HDF5 record each
  (`<piece>_s<k>_r<j>`, JPEG q95). Labels = GAPS MIDI shifted by the measured per-piece sync lag
  (`baselines.clips.measure_sync`), cut to each record, as V2N TSV (onset, key_off, pedal_off, note, velocity
  dummy). Writer `src/guitar2midi/baselines/v2n_data.py`. Checks: label sheet (strip frames + roll for 3
  records), drift sheet (tracking over minutes, frames every ~30 s), hours kept per split. Disk: ~25 GB per strip
  version (107 GB free on Nov 27).
- **17c — training wrapper + sanity.** (Built Nov 27: batch 8 × gradient accumulation 2 instead of 16, best checkpoint by val note onset F1; all checks passed, see log.) `35c_train_v2n.py` (env `v2n`): Hydra cfg = V2N `oofv_1s` + our overrides
  (`configs/model/v2n_guitar/`), optional piano init, calls V2N's trainer, run dir with merged config. Checks:
  unit test MIDI → TSV → `LinearLabeler` round trip and HDF5 readable by `VPTDataset`; load check (key count);
  overfit 1 record ~300 steps → train onset F1 near 100 %; generalised `V2NTranscriber` with the original config
  still reproduces the CP6 outputs.
- **17d — runs A, D, B, C.** Curves (train/val loss, val onset F1).
- **17e — evaluation.** `30g_run_v2n.py --config --checkpoint --system v2n_ft_X` on the existing test strips of
  GAPS (74), GuiFiR-Vid (32), magcil (23) (D on `strips_full/`); `30e` masked with other-video control;
  `31a_check_guifir_tab`; GAPS seen/unseen performers; rows in `30i` results table; piano rolls 3 clips per set.

## Reading the results

GAPS = main result (real MIDI). GuiFiR-Vid/magcil rows are pseudo-GT cross-dataset rows: compare systems with each
other and with chance, not with 100 % (the reference itself finds 70 % of GuiFiR's manual tab notes). A large drop
from GAPS to GuiFiR/magcil means GAPS-specific appearance was learned (nylon strings, concert video), not guitar
playing in general.
