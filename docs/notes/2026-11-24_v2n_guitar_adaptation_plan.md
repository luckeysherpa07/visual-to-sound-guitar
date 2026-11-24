# V2N on guitar — what to change (draft plan for CP8, 2026-11-24; now CP12 after the renumbering)

Status: **draft, not approved yet.** Written after CP6 (V2N zero-shot = silence). To be reviewed before CP8 starts.


## Context

CP6 showed that the released V2N checkpoint outputs silence on guitar strips (1 note in 74 clips). The question
now: what has to change so the V2N *architecture* can be trained on guitar input?

**Correction on the "foundation model" point.** The released V2N does **not** use a pretrained foundation
model. Its default feature extractor (`configs/oofv_1s/model/default.yaml`) is `S2SVisualFeatureExtractor`:
a ResNet-18 trained **from scratch** on PianoVAM (`pretrained=True` raises an error, `model/module.py:150`).
Every weight in `v2n_pianovam.safetensors` has only ever seen one piano. The code does contain a second,
generic option: `VisualFeatureExtractor` with timm `convnext_tiny.fb_in22k` (ImageNet-22k pretrained,
`model/module.py:33`), which is commented out in the config. That option is the "foundation" backbone the user
probably saw, and it is a good candidate for guitar.

**How piano-specific the architecture is** (from reading the V2N code, commit a6408c4):

| Part | Piano-specific? | Change for guitar |
|---|---|---|
| Input 800×144 grayscale, 25 fps, 1 s windows | no | none: our CP5 strips already match |
| S2S ResNet-18 / ConvNeXt-T backbone | no (only the *trained weights* are piano) | retrain |
| Slope prior (`_make_slope_vector`, 88-step ramp → upsampled to feature width) | weakly: it is just a left→right position code | keep; with nut-left strips it encodes "position along the neck" |
| Conformer backbone + BiLSTM heads | no | keep |
| `num_classes=88`, `min_midi=21` | output size only | keep 88 (guitar E2–B5 = MIDI 40–83 lies inside; unused outputs learn "never on"); lets the piano heads be warm-started without surgery |
| Offset + velocity heads | yes (key release, Disklavier velocity) | drop: use V2N's existing `Transcriber_OnsetFrame` (Conformer + BiLSTM, onset + frame only); loss weights onset 2, frame 1 |
| Labels: TSV `onset, key_off, pedal_off, note, velocity` → `LinearLabeler` | format only | convert GAPS MIDI to this TSV (key_off = pedal_off = note end, velocity dummy) |
| Data: HDF5 of JPEG frames per record + `tsv_path` attr | format only | write our strips into the same HDF5 format, then V2N's `train.py` runs unchanged |
| Augmentation (`AlbumentationsVideoTransform`: brightness, small rotation, erase, noise) | no | keep; no horizontal flip (would mirror pitch) |

So **no change to V2N's model code is needed**: guitar training = new data in V2N's format + a Hydra config
that picks `Transcriber_OnsetFrame` and the backbone.

## Approach (recommended)

### 1. Training strips for GAPS train split (`scripts/30h_make_train_strips.py`)
- Download GAPS train videos/audio/MIDI: existing `scripts/30a_download_gaps.py --split train`.
- Pick one frame per video, click nut + fret 12 with existing `scripts/01b_annotate_corners.py`
  (~270 clicks ≈ 1 h; saved as `data/labels/fretboard_corners/gaps_train.jsonl`, allowed as future FPE training data).
- Reuse CP5 code: `preprocessing/video.py` (`detect_cuts`, `shots`), `preprocessing/fretboard_tracking.py`
  (`track_corners`, `warp_strip`) to track through the clicked shot of the **whole performance** (not 8 s clips).
- Only frames inside the clicked shot are kept; each contiguous valid run ≥ 2 s becomes one HDF5 record
  `<piece>_s<k>`, so V2N never trains on black/no-view frames.
- Labels: GAPS MIDI shifted by the per-piece sync lag (`baselines.clips.measure_sync`), cut to each record,
  written as V2N TSV. Performer-level hold-out of ~10 % of train pieces as validation (CLAUDE.md: split by performer).
- Output: `data/processed/v2n_guitar/processed_v2/{train,valid}/guitar_v2.hdf5` + `TSV/` (V2N's layout,
  JPEG q95 as in `data/preprocess_video.py`). Writer in `src/guitar2midi/baselines/v2n_data.py`.

### 2. Configs (`configs/model/v2n_guitar/`, Hydra, overriding V2N's `oofv_1s`)
- `transcriber: model.transcriber.Transcriber_OnsetFrame`, `num_classes: 88`, `dim: 512`.
- `train.loss.weights: {onset: 2, frame: 1}`; `report_to: [none]` (no wandb); `bf16`, `torch_compile` as V2N.
- `max_steps` ~20k per run (V2N: 100k ≈ 2 days on RTX 5080; we have ~4 days for CP8 on an RTX 4080).
- Eval callback: same 50 ms onset tolerance.

### 3. Three runs (CP8 rows of the CP7 table)
| run | backbone | init | why |
|---|---|---|---|
| A (main) | S2S ResNet-18 | V2N piano checkpoint, full fine-tune (onset/frame heads + backbone loaded, offset/velocity weights discarded) | does piano visual knowledge transfer? |
| B | ConvNeXt-T `fb_in22k` | ImageNet-22k (generic "foundation" features) | generic pretraining vs piano pretraining |
| C | S2S ResNet-18 | scratch | lower bound: what guitar data alone gives |
Loading for A: `load_state_dict(strict=False)` after filtering `offset_head.*`/`velocity_head.*`; assert all
other keys load. Run order A → B → C; C dropped first if time runs out (CLAUDE.md fallback: no results after ~4
days → move to main experiments).

### 4. Training entry point (`scripts/30h_finetune_v2n.py`, env `v2n`)
Thin wrapper: builds the Hydra cfg (V2N `oofv_1s` + our overrides), optional init checkpoint, calls V2N's
`train.train(cfg)`; output dir `outputs/runs/<date>_cp8_v2n_<A|B|C>/` with merged config copied in.

### 5. Evaluation on the CP6 test clips
Generalise `guitar2midi.baselines.v2n.V2NTranscriber` to take a model config + checkpoint (currently hard-wired
to `oofv_1s` / `Transcriber_OnsetOffsetFrameVelocity`); `scripts/30g_run_v2n.py --config --checkpoint --system
v2n_ft_A` → `transcriptions/v2n_ft_A/`; then `30e_evaluate_baselines.py --mask-dir .../strips` with all rows.
Test clips stay EVAL_ONLY (GAPS test split never used for training).

## Files

- New: `src/guitar2midi/baselines/v2n_data.py` (strips + MIDI → V2N HDF5/TSV), `scripts/30h_make_train_strips.py`,
  `scripts/30h_finetune_v2n.py`, `configs/model/v2n_guitar/*.yaml`, `tests/test_v2n_data.py`.
- Changed: `src/guitar2midi/baselines/v2n.py` (configurable model/checkpoint), `scripts/30g_run_v2n.py`
  (`--config/--checkpoint/--system`), `CLAUDE.md`, `docs/notes/baseline_study_log.md`.
- V2N repo stays unmodified (pinned a6408c4).

## Verification

1. Unit test: MIDI → TSV → V2N `LinearLabeler` round trip gives onsets at the right frames/pitches; HDF5 record
   readable by V2N's `VPTDataset` (len, shapes `(25,144,800)` and labels `(25,88)`).
2. Visual check: label sheet (strip frames + piano roll from the HDF5 labels) for 3 train records, sync visible.
3. Overfit sanity: run A on 1 record for ~300 steps → train onset F1 near 100 % (pipeline can learn).
4. Load check for A: every non-offset/velocity key of the piano checkpoint is loaded (printed count).
5. After training: piano rolls (existing `v2n.plot_roll`) for the 3 CP6 clips + table rows A/B/C vs zero-shot,
   MMAudio, ceiling on the masked 74 clips.
