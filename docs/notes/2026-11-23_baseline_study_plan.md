# Plan: Baseline check — how bad are existing video→audio models on guitar?

## Context

Supervisor meeting (2026-11-23): pause the GuiFiR FPE / hand-pose work. It goes into fine detail before we have a
bird's-eye view. First answer: *if we run existing off-the-shelf video-to-audio (V2A) models on our guitar videos,
how wrong are the notes?* This gives the thesis its motivation and a baseline row. Afterwards we resume GuiFiR (with
hand pose); the FPE plan is kept.

**Core idea:** silent video → V2A model → generated audio → audio transcription → MIDI → compare with GAPS
ground-truth MIDI (mir_eval note F1, same metric as the thesis). Transcribing the **real** audio with the same model
is the ceiling, so the gap is purely the V2A error.

**Models:** MMAudio (primary, open weights, fits RTX 4080 16 GB). Optional if time allows: HunyuanVideo-Foley /
ThinkSound (if they fit), Foley Music (if weights exist). Piano-only models (Audeo, S2S, V2N) and video-to-music
models are related work only. No public guitar video→MIDI model exists; that gap is the thesis.

## Checkpoints

Each checkpoint ends with something you can **see** and a short stop where you check it before we continue.

### CP0 — Save the plan in CLAUDE.md (before any code)
- CLAUDE.md: status line (FPE paused after supervisor meeting 2026-11-23, hand pose afterwards); new section
  "Phase: baseline study (existing models)" with the CP1–CP9 list, each with a status marker (todo / in progress / done);
  decision-log rows (baseline study first; V2N zero-shot with manual-corner crops; V2N fine-tune = full fine-tune,
  LoRA only as an optional ablation); planned `30*` scripts in the project structure; timeline note.
- Also copy the full plan to `docs/notes/2026-11-23_baseline_study_plan.md`.
- **You see:** the CLAUDE.md diff. Stop for your review; commit only if you say so.

### CP1 — GAPS data in place
- Unzip `~/Downloads/gaps_v1_no_audio.zip` → `data/raw/gaps/`; download audio (HF `xavriley/GAPS`) and test-split videos (yt-dlp).
- Code: `src/guitar2midi/data/gaps.py` (metadata, splits, yt_id ↔ MIDI ↔ audio), `scripts/30a_download_gaps.py`.
- **You see:** a summary printout: N test pieces, how many videos downloaded / unavailable, total hours.

### CP2 — Evaluation clips
- Pick ~20 test pieces × 3 clips × 8 s where the guitar is visible. Cut silent video + real audio + GT MIDI per clip.
  Check video↔audio sync (cross-correlation); drop clips off by more than 1 frame.
- Code: `src/guitar2midi/baselines/clips.py`, `scripts/30b_make_eval_clips.py` → `data/interim/baseline_clips/`.
- **You see:** `clips.csv` + a contact sheet image (one frame per clip) + 3 clips with GT MIDI rendered as a piano roll.

### CP3 — Transcription ceiling (audio → MIDI works)
- Transcribe the **real** audio with the GAPS authors' guitar model (HF `xavriley/midi-transcription-models`; fallback basic-pitch).
- Code: `src/guitar2midi/evaluation/note_metrics.py` (note/frame F1, chroma similarity, onset correlation) + `tests/test_note_metrics.py`; `scripts/30d_transcribe.py`.
- **You see:** ceiling note F1 on our clips. It should be near the GAPS paper's numbers. If not, the alignment is broken; fix before CP4.

### CP4 — MMAudio runs on guitar video
- Separate conda env `mmaudio` (needs newer torch than our pinned 2.1.2). Generate audio for all clips, without a prompt and with "solo classical guitar".
- Code: `scripts/30c_run_mmaudio.sh`.
- **You see:** ~5 example videos with the generated audio muxed in, to listen to (and show the supervisor).

### CP5 — Fretboard strips for V2N (cropping)
- Click 4 corners (nut + fret 12) on one frame per clip with the existing `scripts/01b_annotate_corners.py`
  → homography (`fretboard_grid.corners_from_clicks`) → 800×144 grayscale strip, same as V2N's input. One warp per
  8 s clip (static GAPS cameras). Fair, exact crop, so V2N's failure comes from piano→guitar, not from bad cropping.
  The clicks also become GAPS training frames for the FPE later.
  (GuiFiR FPE crop not possible yet: no released weights, our FPE untrained, no neck detector for GAPS.)
- Code: `src/guitar2midi/baselines/strips.py`, `scripts/30f_make_v2n_strips.py` → `data/interim/baseline_clips/strips/`.
- **You see:** a grid of all warped strips (one frame each), to check every crop before running V2N.

### CP6 — V2N zero-shot (video → MIDI directly, piano-trained)
- Separate env `v2n`; run the public checkpoints (github.com/yonghyunk1m/V2N) on the strips. Output MIDI is scored directly (no transcription step).
- Code: `scripts/30g_run_v2n.sh`.
- **You see:** V2N's predicted piano roll next to the GT roll for 3 clips.

### CP7 — Results table ("how bad")
- Transcribe the generated audio; evaluate all rows: ceiling / MMAudio no prompt / MMAudio + prompt / V2N zero-shot / audio from a different clip (does the video matter?) / silence.
- Code: `scripts/30e_evaluate_baselines.py` → `outputs/runs/<date>_v2a_baseline/` + `thesis/tables/`.
- **You see:** one table (note F1 @50/100 ms, frame F1, chroma similarity, onset correlation) + a one-paragraph summary for the supervisor.

### CP8 — V2N fine-tuning pilot on guitar
- Strips for GAPS **train** split: click corners once per *video* (static cameras; ~200 clicks ≈ 1 h), flag videos
  where the guitar moves (corner check at a few frames). No FPE needed.
- Model: V2N with the 88-key heads + slope prior replaced by guitar pitch heads (onset + frame, ~E2–B5) and a
  fretboard position encoding. Velocity head dropped.
- Compare: (a) V2N piano weights + **full fine-tune**, (b) same model **from scratch**, (c) optional: LoRA on the
  backbone. Full fine-tune is the main row: V2N is small (ResNet-18 + conv blocks + BiLSTM, no attention) and fits
  in 16 GB, so LoRA saves nothing important and fits conv/LSTM layers poorly.
- Evaluate on the CP2 test clips; rows added to the CP7 table.
- Code: `src/guitar2midi/model/` (V2N adaptation, reused later for the main model), `scripts/30h_finetune_v2n.py`.
- **You see:** training curves + table rows: zero-shot vs fine-tuned vs scratch.

### CP9 (optional) — Extend
- More V2A models (HunyuanVideo-Foley / ThinkSound / Foley Music) if they run; magcil clips (labelled **pseudo**, compared with transcription of the real audio, no headline numbers); own data later with the same scripts.
- **You see:** extra rows in the same table.

After CP7: update CLAUDE.md and commit (the "how bad" result is ready for the supervisor). After CP8/CP9 again: update CLAUDE.md (paused FPE, new phase, scripts, envs, decision log) and commit.

## Verification
- Unit tests: GT vs itself → F1 = 1; silence → F1 = 0.
- CP3 ceiling matches the published GAPS level.
- Listen to generated clips before trusting the numbers.

## Rough effort
CP1–CP2: ~half a day (mostly downloads). CP3: ~half a day. CP4: ~half a day. CP5: ~2 h + 15 min clicking. CP6: ~half a day. CP7: a few hours. CP8: ~3–4 days (1 h clicking, model adaptation, training runs). CP9: open.
