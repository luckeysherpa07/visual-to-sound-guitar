# How well do existing models transcribe silent guitar video? — Baseline study summary

Luckey Sherpa, 2026-11-24 (for the meeting of 23 Nov: "how bad are existing models, really?")

## Short answer

Badly. No existing model recovers the played notes from silent guitar video. The best one (MMAudio, video →
audio → transcription) gets **3–7 % note F1**, at the level of a random clip of another piece. The closest visual
transcription model (V2N, piano) outputs **nothing at all**. The only thing that is recovered better than chance
is **rhythm on GAPS** (timing F1 36 vs 26 for the wrong video), and even that disappears on the two other datasets.
The same transcription pipeline on the real audio reaches 94 %, so the gap comes from the video→sound step, not
from our evaluation.

## Setup

- **Task:** silent guitar video → note events (MIDI). Metric: mir_eval note F1 (onset ±50 ms), plus frame F1,
  timing F1 (onsets only, pitch ignored), chroma similarity and onset-curve correlation.
- **Systems:**
  - *MMAudio* (CVPR 2025, open weights) generates audio from the silent clip, with and without a text prompt
    ("solo classical / acoustic / electric guitar"); the audio is transcribed with the GAPS benchmark guitar model.
  - *V2N* (ISMIR 2026, visual piano transcription, public checkpoint) on fretboard strips that we crop exactly
    like its piano input (manually clicked / GT corners, tracked through each clip, 800×144 grayscale).
  - Controls: the same transcription on the **real audio** (ceiling), MMAudio with the **video of another clip**,
    real audio of **another piece** (chance level), **silence**.
- **Data** (clips of 8 s, or the whole recording if shorter (magcil bends ≈ 3 s), at 25 fps; only frames with a visible fretboard are scored, the same for every system):

| Dataset | Guitar / video | Clips | Labels |
|---|---|---|---|
| GAPS (test split) | classical, YouTube concert videos | 74 | real MIDI |
| GuiFiR-Vid (test split) | steel-string acoustic, YouTube tutorial videos | 32 | pseudo-GT (transcription of the real audio); manual tablature at ~20 frames/clip as an independent check |
| magcil | electric (3 guitars × 3 amps), 9 playing techniques, smartphone | 23 | pseudo-GT (transcription of the real audio) |

On GuiFiR-Vid and magcil there is no real MIDI. The pseudo-GT is checked: it finds 70 % of GuiFiR's manual tab
notes, and 98 % of its notes on magcil use the pitches of the written exercises (chance 58 %).

## Results (note F1 @ 50 ms, %; full table: `thesis/tables/baseline_results.tex`)

| | GAPS | GuiFiR-Vid | magcil |
|---|---|---|---|
| Real audio (ceiling / reference) | **93.6** | (reference) | (reference) |
| MMAudio, video only | 3.1 | 2.7 | 5.9 |
| MMAudio, video + prompt | 3.0 | 2.8 | 6.6 |
| V2N zero-shot | 0.0 | 0.0 | 0.0 |
| MMAudio with the video of another clip | 1.6 | 1.7 | 3.2 |
| Chance (real audio of another piece) | 2.1 | 1.6 | 4.3 |
| Timing F1, MMAudio vs mismatched video | **35.5 vs 25.8 (p < 0.001)** | 27.5 vs 26.7 (n.s.) | 46.8 vs 44.0 (n.s.) |
| GuiFiR manual tab notes found | – | MMAudio 6 %, chance 3 %, real audio 70 % | – |

## What we learned

1. **Video-to-audio models make guitar *sound*, not guitar *notes*.** MMAudio's output sounds like the right
   instrument (plucked nylon, steel-string, distorted electric) with a plausible rhythm, but the pitches are
   essentially random. The prompt changes nothing.
2. **Some rhythm is visible to MMAudio, but only in favourable videos.** On GAPS (concert videos, hands large in
   the frame) its onsets follow the real plucks significantly better than with a wrong video. On tutorial videos
   (small hands, overlays) and fast electric playing (8–10 notes/s) this signal is gone.
3. **Visual piano transcription does not transfer to guitar.** V2N outputs silence on all three datasets. We
   checked this carefully: the wrapper reproduces V2N exactly on piano (94 % F1), and 16 input variants (flips,
   a piano-like crop layout, brightness matched to the piano frames) × 3 datasets give no notes, and no timing or
   pitch correlation even below the decision threshold.
4. **The bottleneck is the visual model, not the evaluation.** The transcription model reaches 93.6 % on the real
   GAPS audio (paper: 93.4–94.3 %), so the pipeline and metrics are sound.

So there is no usable off-the-shelf solution; a guitar-specific visual model is needed. This is the motivation
for the thesis.

## Next steps

- **V2N fine-tuning pilot (~4 days):** keep V2N's architecture, train it on GAPS train-split fretboard
  strips with real MIDI (piano initialisation vs ImageNet init vs scratch). This shows whether the architecture can
  learn guitar at all once it sees guitar data.
- **Picking hand:** the current fretboard strips show the fretting hand (pitch) but not the picking hand (when a
  note sounds; open strings, repeated notes). Before the pilot, a second strip version covering the full string
  (nut → bridge) is made from the same tracking, and one pilot run uses it, to measure how much the picking hand
  helps.
- Then resume the GuiFiR fretboard / hand-pose work for a robust automatic crop (now with hand pose, as discussed).

Caveats: the pseudo-GT datasets are small (32 and 23 clips) and their reference is imperfect (it misses repeated
and slid notes), so their absolute numbers are less precise than GAPS; the conclusion (≈ chance everywhere) does not
depend on that.
