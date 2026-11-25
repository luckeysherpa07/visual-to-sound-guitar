# Supervisor meeting — 2026-11-25

Luckey Sherpa with Zongwei Wu, after the baseline study CP0–CP12 (results table CP11: MMAudio ≈ chance, V2N
zero-shot = silence on GAPS / GuiFiR-Vid / magcil).

## Requests (in this order)

1. **More baselines before going ahead:** more video → audio (V2A) models and video → MIDI (V2M) models.
2. **LoRA fine-tune.**
3. **V2N trained on guitar** (the planned fine-tuning pilot).
4. Then **propose an own model**.

## Decisions after the meeting (Luckey + Claude, 2026-11-25)

- New checkpoint order: CP13 more V2A baselines → CP14 V2M baselines → CP15 LoRA fine-tune of a V2A model →
  CP16 V2N trained on guitar → CP17 results table + summary. The old CP13 (V2N pilot) becomes CP16, the old
  optional CP14 is absorbed into CP13/CP14.
- **Baselines: check first, then pick.** A short availability check (public code, released weights, licence, fits
  the RTX 4080 16 GB, effort) for HunyuanVideo-Foley, ThinkSound, Foley Music, Audeo and 1–2 others; then run
  about 2 V2A and 1–2 V2M models on all three datasets with the existing pipeline.
- **LoRA goes on the V2A model (MMAudio), not on V2N.** LoRA pays off for large pretrained transformers that do
  not fit a full fine-tune in 16 GB and whose general knowledge should be kept. V2N is small (~28 M weights,
  ResNet-18 + conv + BiLSTM, no attention) and has to change a lot from piano keys to guitar frets, so it gets a
  **full fine-tune only** (no V2N LoRA run).
- The CP15 question: trained on GAPS train video + real audio, does MMAudio learn to produce the *right notes*,
  not just guitar-like sound? Scored with the same transcription pipeline. This compares "adapt a large
  foundation model" (CP15) with "train a specialised visual model" (CP16).

## Brainstorm for later (not planned yet)

- **Own model:** after CP16, based on what the baselines and the V2N runs show (e.g. picking-hand stream, hand
  pose, fretboard position prior).
- **Own dataset idea:** same player, same guitar, only the **camera distance and angle** change (left / right /
  top / bottom, near / far). A controlled viewpoint study: how much does each model lose when the view changes?
  Fits the earlier "own multi-view recordings" plan; recording fallback rule (single view if not done by Nov 30)
  still applies.
