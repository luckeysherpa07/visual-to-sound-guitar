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

## Extension (2026-11-24): GuiFiR-Vid and magcil

Before the results table, the same baselines also run on two more guitar datasets, so "how bad are existing models"
covers more than classical concert videos. Checkpoints were renumbered: **one checkpoint per dataset for all of its
preprocessing (CP7, CP8), then one checkpoint per task that runs on all datasets (CP9–CP11)**. CP0–CP6 unchanged;
old CP7 → CP11, old CP8 → CP12, old CP9 → CP13 (later on 2026-11-24 the full-string strips were inserted as CP12, so the fine-tune is CP13 and the optional extensions CP14). Earlier names stay as they are (log section "CP7 (part 1)", run
folder `2026-11-24_cp7_v2n_masked`).

| | GAPS (CP0–6) | GuiFiR-Vid (CP7) | magcil guitar_style_dataset (CP8) |
|---|---|---|---|
| Guitar / music | classical nylon, classical repertoire | mostly steel-string acoustic + some classical; pop/anime/film covers | electric, 3 guitars × 3 amps; 9 technique exercises (alternate/sweep picking, bend, hammer-on, pull-off, legato, slide, tapping, vibrato) |
| Video | YouTube concerts, mostly 720p, multi-camera cuts | YouTube tab-tutorial clips, 10 s, mostly 640×360 @ 29.97, tab overlays, front camera | own recordings, mp4 (view/fps to check) |
| Audio | yes (HF) | yes, inside the mp4 | separate .wav |
| Note labels | **real MIDI** | sparse tablature (~20 frames/clip, fret per string + capo, GT or audio-derived), no onsets/offsets | none time-aligned; MuseScore scores of the exercises |
| Fretboard corners | clicked (eval only) | GuiFiR GT corners on test frames → no clicking | clicked |
| Size used | 90 clips (74 with strips) | ≤ 32 clips (1 per test song) | 27 clips (3 per technique) |

Decisions (Luckey, 2026-11-24):
- **Pseudo-GT in the same table.** GuiFiR and magcil have no real MIDI; they are scored against the GAPS model's
  transcription of their real audio. All three datasets go into the same CP11 table, with a label-source column
  (real MIDI / pseudo-GT) and a footnote. The ceiling row is 100 % by construction there, so the tab check (GuiFiR)
  and a spot check (magcil) report how trustworthy the reference is. Trained-model main results (CP13+) stay on real MIDI; GuiFiR-Vid / magcil only as secondary, labelled cross-dataset rows.
- **GuiFiR tab = check + extra column.** (a) How often the reference contains the tab pitch at tab frames (CP7);
  (b) a column "pitch recall at tab frames" for every system in the GuiFiR block. Recall only: the tab is sparse.
- **Systems:** MMAudio (± prompt) and V2N zero-shot on both new datasets.

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
  ~~The clicks also become GAPS training frames for the FPE later.~~ Corrected Nov 23: these are GAPS *test*
  clips, so their clicks are evaluation-only (FPE training on them would leak into the test evaluation).
  Added Nov 23: corners are tracked through each clip (zoom/pan/guitar movement/cuts), see the progress log.
  (GuiFiR FPE crop not possible yet: no released weights, our FPE untrained, no neck detector for GAPS.)
- Code: `src/guitar2midi/baselines/strips.py`, `scripts/30f_make_v2n_strips.py` → `data/interim/baseline_clips/strips/`.
- **You see:** a grid of all warped strips (one frame each), to check every crop before running V2N.

### CP6 — V2N zero-shot (video → MIDI directly, piano-trained)
- Separate env `v2n`; run the public checkpoints (github.com/yonghyunk1m/V2N) on the strips. Output MIDI is scored directly (no transcription step).
- Code: `scripts/30g_run_v2n.py` (+ positive control `scripts/30g_check_v2n_piano.py`).
- **You see:** V2N's predicted piano roll next to the GT roll for 3 clips.

### CP7 — GuiFiR-Vid preprocessing (added 2026-11-24)
- Eval clips from the GuiFiR **test** split: 1 clip per song (≤ 32 clips), 8 s, 25 fps; silent video + real audio
  (GuiFiR mp4s contain audio) → `data/interim/baseline_clips_guifir/`, same layout as the GAPS clips.
- Reference transcription of the real audio (GAPS model, env `amt`, `30d_transcribe`) = pseudo-GT.
- Tab check: GuiFiR `frame_tab` (fret per string, 20 = not played) + `frame_capo` + standard tuning → pitch at
  each annotated frame; report how often the reference contains it (reliability of the pseudo-GT).
- V2N strips **without clicking**: seed `fretboard_tracking.track_corners` from the GuiFiR GT corners of the first
  annotated frame in the window (`rb, rt, lb, lt`, × frame_width/640). Test corners = evaluation only, never FPE training.
- Code: `src/guitar2midi/baselines/clips_guifir.py` (reuses `baselines/clips.py`, `data/guifir.py`), tab→pitch helper,
  `scripts/31a_make_guifir_clips.py`.
- **You see:** clips.csv, contact sheet, tab-agreement %, tracking sheet.

### CP8 — magcil preprocessing (added 2026-11-24)
- No download: extract `~/Downloads/guitar_style_dataset-v1.0.0.zip` (14.4 GB, commit `eb27d7b`) unchanged into
  `data/raw/magcil/` (+ `SOURCE.txt`); copy the dataset paper
  (`A_multimodal_dataset_for_electric_guitar_playing_t.pdf`) to `docs/papers/`.
- Reader `src/guitar2midi/data/magcil.py` (technique, guitar, amp, video, wav). Check camera view / resolution / fps
  and video↔wav sync (`baselines.clips.measure_sync`).
- Eval clips: 3 per technique (27 clips, 8 s, 25 fps), `scripts/31b_make_magcil_clips.py`.
- Reference transcription + spot check (electric guitar through amps vs a classical-trained model).
- V2N strips with the existing 30f tools (you click nut + fret 12 per clip).
- **You see:** summary, contact sheet, 3 piano rolls, tracking sheet.

### CP9 — MMAudio on all datasets (added 2026-11-24)
- `30c_run_mmaudio` unchanged (large_44k_v2, seed 42), no prompt and "solo classical / acoustic / electric guitar"
  per dataset; `30d` transcribe; `30e` evaluate vs GT (GAPS) or reference (GuiFiR, magcil). GAPS outputs from CP4 reused.
- Scripts get a `--clips-dir` argument where the GAPS path is hard-coded.
- **You see:** spectrograms, rows per dataset.

### CP10 — V2N zero-shot on all datasets (added 2026-11-24)
- `30g_run_v2n` on the GuiFiR and magcil strips (GAPS from CP6), plus the open CP6 re-check of the 0.0 result
  (flip / crop layout / thresholds as a diagnostic) on all three datasets.
- **You see:** predicted vs reference rolls, rows per dataset.

### CP11 — Results table ("how bad") — was CP7 before the 2026-11-24 renumbering
- One table with a block per dataset (GAPS real MIDI, GuiFiR-Vid pseudo-GT, magcil pseudo-GT; label-source column), GuiFiR tab-recall column. Transcribe the generated audio; evaluate all rows: ceiling / MMAudio no prompt / MMAudio + prompt / V2N zero-shot / audio from a different clip (does the video matter?) / silence.
- Code: `scripts/30e_evaluate_baselines.py` → `outputs/runs/<date>_v2a_baseline/` + `thesis/tables/`.
- **You see:** one table (note F1 @50/100 ms, frame F1, chroma similarity, onset correlation) + a one-paragraph summary for the supervisor.

### CP12 — Full-string strips nut → bridge (added 2026-11-24)
- Why: the current strips (nut → fret 20) show the fretting hand, which mostly gives the pitch. *When* a note sounds is
  decided by the picking hand over the soundhole / near the bridge, which is outside the strip; open strings, repeated
  notes and which strings of a held chord are plucked cannot be seen from the fretting hand. (Piano differs: the same
  finger chooses the pitch and starts the note.)
- What: a second strip version covering the full string length, nut → bridge, for the test clips of all three
  datasets (GAPS 74, GuiFiR-Vid 32, magcil 23). The bridge is 2 × the nut–fret-12 distance along the neck (fret 12
  = half the scale length), mapped with the same tracked corners/homography → no new clicks, same tracking. 800×144
  as before (each fret gets fewer pixels). GAPS/magcil via `30f_make_v2n_strips.py`, GuiFiR via `31a_make_guifir_clips.py`
  (option for the strip extent; default output unchanged, new strips in `strips_full/`).
- Check: tracking sheet with both strip versions. The bridge part is extrapolated beyond the clicked points (less
  precise; perspective) and can be outside the frame in close-ups (black). Valid-frame masks stay the same as for
  the fretboard strips, so old and new rows are compared on the same frames.
- V2N zero-shot (released piano model, no training) on the **full-string (nut → bridge) strips**, one row per dataset,
  next to the existing zero-shot rows on the fretboard strips (nut → fret 20). Expected silent like all CP10 variants;
  it completes the picture. The real picking-hand test is CP13 run D (trained on full-string strips) vs run A.
- Evaluation code, references, masks, results table unchanged; MMAudio and the existing V2N results are unaffected.
- **You see:** tracking sheet (fretboard strip vs full-string strip per clip), V2N zero-shot rows on the full-string
  strips (vs the fretboard-strip rows).
- Effort: ~half a day. Also prepares the CP13 training strips (run D).

### Extension (2026-11-25): more baselines and LoRA before the V2N training — supervisor meeting
Supervisor (meeting 2026-11-25, notes: `2026-11-25_supervisor_meeting.md`): first more video → audio and video →
MIDI baselines, then a LoRA fine-tune, then V2N trained on guitar, then an own model. New order: CP13 V2A
baselines, CP14 V2M baselines, CP15 LoRA (MMAudio), CP16 V2N trained (was CP13), CP17 table + summary. The old
optional CP14 is absorbed into CP13/CP14. LoRA only on the V2A model; V2N gets a full fine-tune only.

**Renumbered 2026-11-26 (Luckey):** the V2M baselines are split into two checkpoints, CP14 Audeo zero-shot and
CP15 Foley Music (retrained on its own data), because they are independent and differ in risk. Everything after
shifts by one: CP16 LoRA (MMAudio, was CP15), CP17 V2N trained (was CP16), CP18 table + summary (was CP17).

### CP13 — More video → audio baselines, all datasets (added 2026-11-25)
- Step 1 (stop for review): availability check for V2A **and** V2M candidates: HunyuanVideo-Foley, ThinkSound,
  Foley Music, Audeo + 1–2 others; per model: public code, released weights, licence, fits 16 GB (or with
  offloading / fp16), input requirements, rough effort. Then pick ~2 V2A models (this CP) and 1–2 V2M models (CP14/CP15).
- Step 2: each chosen V2A model in its own conda env (`envs/<name>.yml`), same as MMAudio: silent clips of GAPS /
  GuiFiR-Vid / magcil → audio (± the same text prompts) → `30d_transcribe` → `30e_evaluate_baselines` (masked,
  shuffled-video control, GuiFiR tab recall).
- **You see:** availability table, spectrograms / listening examples, rows per dataset.

### CP14 — Audeo zero-shot, all datasets (added 2026-11-25 as part of the V2M CP, own CP since 2026-11-26)
- Audeo (NeurIPS 2020): piano top-view video → Video2RollNet (piano roll) → Roll2Midi (GAN refinement) → MIDI.
  Public weights (Google Drive; download with gdown, verify). No training: run on our fretboard strips like V2N
  zero-shot (resize/crop to Audeo's input as its own preprocessing does).
- Output is MIDI, so scored directly with `30e` (masked, other-video control) + GuiFiR tab recall. Expected near
  silence (piano model, like V2N); a second piano V2M row shows the V2N result is not specific to one model.
- If the weights cannot be downloaded: related work only, with the reason.
- **You see:** piano rolls, rows per dataset. Effort ~half a day.

### CP15 — Foley Music, all datasets (added 2026-11-25 as part of the V2M CP, own CP since 2026-11-26)
- Foley Music (ECCV 2020): body + hand keypoints → MIDI events (Graph-Transformer). No released checkpoint, so
  **retrained from scratch with the authors' code on their own released data** (`data_pose_midi.tar`, MUSIC
  `acoustic_guitar` config). Never trained on our data, so it is still a zero-shot baseline for our three sets.
- Sanity check: our retrained model vs the paper's numbers on their data (it is a reproduction, not the authors'
  model; say so in the thesis).
- Keypoints for our test clips with the same kind of pose detector as theirs (likely OpenPose; check the repo),
  then MIDI → `30e` (masked, other-video control) + GuiFiR tab recall.
- Time box ~1–2 days (2020 code, PyTorch 1.4, own env `foleymusic`): if it does not install/train by then,
  related work only, with the reason.
- **You see:** training curve, piano rolls, rows per dataset.

### CP16 — LoRA fine-tune of a V2A model (MMAudio) on guitar (added 2026-11-25 as CP15, renumbered 2026-11-26)
- MMAudio (already installed, env `mmaudio`) + LoRA adapters on its transformer; base weights frozen. Training
  data: GAPS train split (267 pieces with video, 14.7 h): 8 s windows of silent video + real audio, sync-corrected;
  ~10 % of pieces held out by performer as validation.
- Evaluated exactly like CP9 on all three test sets (generate → transcribe → evaluate), compared with MMAudio
  zero-shot. Question: does it learn the right notes, or only better guitar timbre?
- Detailed plan (LoRA rank / target layers, steps, memory) written before starting, like the V2N plan.
- **You see:** training curves, spectrograms before/after, rows per dataset.

### CP17 — V2N trained on guitar (full fine-tune, no LoRA) — was CP8, CP12, CP13, then CP16 (renumbered 2026-11-25 and 2026-11-26)
- Strips for GAPS **train** split: click corners once per *video* (static cameras; ~200 clicks ≈ 1 h), flag videos
  where the guitar moves (corner check at a few frames). No FPE needed.
- Model: V2N with the 88-key heads + slope prior replaced by guitar pitch heads (onset + frame, ~E2–B5) and a
  fretboard position encoding. Velocity head dropped.
- Compare: runs A (piano init, full fine-tune), B (ConvNeXt-T ImageNet-22k), C (scratch), D (= A on full-string
  strips). **No LoRA on V2N** (decided 2026-11-25): V2N is small (ResNet-18 + conv blocks + BiLSTM, no attention)
  and fits in 16 GB, so LoRA saves nothing important and fits conv/LSTM layers poorly; LoRA is done on MMAudio (CP16).
- Evaluate on the existing strips of all three baseline sets: GAPS test clips (main, real MIDI) and, as unseen
  cross-dataset tests, GuiFiR-Vid and magcil (pseudo-GT + GuiFiR tab recall); rows added to every block of the
  CP11 table. Training uses GAPS train only. Details: `2026-11-24_v2n_guitar_adaptation_plan.md` (updated 2026-11-24).
- Code: `src/guitar2midi/model/` (V2N adaptation, reused later for the main model), `scripts/30h_finetune_v2n.py`.
- **You see:** training curves + table rows: zero-shot vs fine-tuned vs scratch.

### CP18 — Results table + summary, all systems (added 2026-11-25 as CP17, renumbered 2026-11-26)
- `30i_results_table.py` extended with the CP13–CP17 rows (V2A, V2M (Audeo, Foley Music), MMAudio + LoRA, V2N A–D) in every dataset
  block; updated supervisor summary.
- **You see:** the full table.

~~CP14 (optional) — Extend~~: absorbed into CP13/CP14 on 2026-11-25 (more V2A models → CP13, Foley Music → CP14, since 2026-11-26 CP15).
Own data / own model: brainstorm after CP18 (see `2026-11-25_supervisor_meeting.md`).

After CP11: update CLAUDE.md and commit (the "how bad" result is ready for the supervisor). After each later CP: update CLAUDE.md and the log, commit when asked.

## Verification
- Unit tests: GT vs itself → F1 = 1; silence → F1 = 0.
- CP3 ceiling matches the published GAPS level.
- Listen to generated clips before trusting the numbers.

## Rough effort
CP1–CP2: ~half a day (mostly downloads). CP3: ~half a day. CP4: ~half a day. CP5: ~2 h + 15 min clicking. CP6: ~half a day. CP7 (GuiFiR prep): ~half a day. CP8 (magcil prep): ~half a day + ~15 min clicking. CP9/CP10: a few hours each. CP11: a few hours. CP12 (full-string strips): ~half a day. CP13 (V2A baselines): ~2–3 days incl. availability check. CP14 (Audeo): ~half a day. CP15 (Foley Music): ~1–2 days, time-boxed. CP16 (MMAudio LoRA): ~3–4 days. CP17 (V2N trained, was CP13/CP16): ~3–4 days (1 h clicking, model adaptation, training runs; + ~1 day for run D). CP18: a few hours.
