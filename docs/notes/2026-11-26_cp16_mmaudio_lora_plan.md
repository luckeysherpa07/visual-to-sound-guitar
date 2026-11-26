# CP16 plan — LoRA fine-tune of MMAudio on GAPS (2026-11-26, approved by Luckey the same day)

Question: if we show MMAudio guitar videos with their real audio, does it learn to play **the right notes**
for the video, or only to **sound more like a classical guitar**?

The boxes marked *Learn* explain the training concepts as they come up (Luckey asked to learn along the way).

---

## 1. What MMAudio is and what "training" it means

> **Learn — what a model "knows".** A neural network is a big function with adjustable numbers (*parameters*,
> also called *weights*). MMAudio large has **1.03 billion** of them. "Training" means: show the model an input,
> compare its output with the correct answer, measure how wrong it is with one number (the **loss**), and nudge
> every weight a little in the direction that makes the loss smaller (**gradient descent**). One nudge = one
> **step**; how big the nudge is = the **learning rate**. One pass over all training examples = one **epoch**.

How MMAudio makes audio (this decides what we train):

1. **Video features** are computed by two frozen helper networks: CLIP (what is in the picture, 8 frames/s) and
   Synchformer (motion/timing, 24 frames/s). Optional text prompt → text features.
2. Audio is not generated as a waveform but as a compressed version (**latent**, 345 steps × 40 numbers for 8 s),
   made by an audio autoencoder (VAE) that turns latents back into sound.
3. The main transformer starts from random noise and, in 25 small steps, turns it into an audio latent that fits the
   video features (**flow matching**: the transformer predicts at every step "which direction leads from noise to
   real audio").

Training it = take a real 8 s clip, add a random amount of noise to its real audio latent, and ask the transformer to
predict the direction back to the real audio, given the video features. Loss = squared error of that direction
(this is exactly MMAudio's own training loss, `FlowMatching.loss`). Nothing about notes appears in the loss: the
model only learns "this video → this sound". Whether that includes the right pitches is what we measure afterwards.

## 2. What LoRA is and why we use it here

> **Learn — full fine-tune vs LoRA.** *Fine-tuning* = continue training a model that is already trained (here on
> VGGSound, ~180k internet videos, plus large audio–text collections) on our own data. A **full fine-tune** changes
> all 1.03 B weights: weights + their gradients + the optimizer's running averages take ~16 GB before any data is
> processed (our GPU has 16 GB), and it easily *forgets* what the model knew, because a few hours of guitar video
> pull everything towards guitar.
>
> **LoRA** (Low-Rank Adaptation, Hu et al. 2021) freezes all original weights and adds a small "correction" next to
> chosen layers. A layer is a matrix **W** (e.g. 896 × 896 numbers); LoRA adds **B·A**, where A is 896 × r and B is
> r × 896 with a small **rank** r (e.g. 16). Output = W·x + B·A·x. Only A and B are trained: 2 × 896 × 16 ≈ 29 k
> numbers instead of 800 k for that layer. B starts at zero, so at step 0 the model is exactly the original MMAudio,
> and training only learns a small change. Afterwards the adapters are a small file (tens of MB) next to the
> unchanged original weights.

Why LoRA on MMAudio but not on V2N (decided Nov 25): MMAudio is big and pretrained on general sound, so LoRA lets
it fit in 16 GB and keeps its general knowledge. V2N is small and must learn new output heads anyway → full
fine-tune in CP17. CP16 vs CP17 = "adapt a big general model" vs "train a small specialised one".

**Where the adapters go:** the linear layers of the 21 transformer blocks (attention q/k/v + output, MLPs). Rank 16
→ roughly 15–20 M trainable numbers (~1.5–2 % of the model). Implementation: a short LoRA wrapper of our own
(~40 lines, `src/guitar2midi/baselines/lora.py`) rather than an extra library, so every line is visible; `peft`
as fallback.

## 3. Data

- **Training:** GAPS **train** split only (267 videos with audio, 14.7 h). Test split never used (as everywhere).
- **Validation split:** ~10 % of the train pieces held out **by performer** (no performer in both parts), so the
  validation loss measures "new player, new piece", not memorising.

> **Learn — why a validation set.** The training loss always goes down: the model can memorise the training clips.
> The **validation loss** on clips it never trains on tells whether it learned something general. When training
> loss keeps falling but validation loss rises, the model **overfits** (memorises). We saw this in CP15: Foley
> Music's validation loss was best at epoch 8 (2.62) and rose to 6.1 by epoch 100. So we keep the checkpoint with
> the best validation loss, not the last one.

- **Windows:** 8 s (MMAudio's training length), every 8 s through each piece, ~6,000 windows. Skip windows where the
  audio is near silent and the 4 videos shorter than their audio past the video end.
- **Audio = the video's own audio track** (the mp4 downloaded from YouTube), not the GAPS wav. This is exactly
  MMAudio's own training recipe (its VGGSound loader reads frames and audio from the same mp4; mono, resampled to
  44.1 kHz; no loudness normalisation, the loader's default), and video and audio are then in sync by construction: no lag measurement or shift
  needed. (The GAPS wav is the same recording but lags the video by 0–49 ms of codec priming; 49 ms is ~1
  Synchformer frame, so using it unshifted would teach a timing offset.) The 4 videos shorter than their GAPS audio
  are no problem this way: windows only cover the video.
- **Whole frame**, not the fretboard strip (MMAudio sees ordinary video). Cuts and close-ups stay in: the model must
  cope with them at test time too.
- **Leakage check (step 16a):** GAPS splits are by piece; check whether any performer of our 30 test pieces also
  plays in train. If yes, report it (the model could learn that player's instrument sound); no change to the test set.

**Precompute features once** (MMAudio's own `FeaturesUtils`, as in its official training pipeline): CLIP, Synchformer,
text, VAE latent mean/std per window → `data/processed/mmaudio_lora/`. Then training never touches video, which
makes each step fast. Estimated 1–2 h for all windows.

## 4. Training runs

| run | input during training | why |
|---|---|---|
| **A (main)** | video features + text ("solo classical guitar", dropped 10 % of the time as in MMAudio) | the LoRA fine-tune |
| **B (control)** | text only, video features always replaced by MMAudio's "empty video" | learns guitar *sound* but cannot use the video. If A ≈ B on note F1, any gain comes from "sounding like guitar", not from watching |

Training loop: MMAudio's own `train_fn` logic (random noise level t from a log-normal, flow-matching loss, video and
text each replaced by the "empty" features 10 % of the time = classifier-free training, needed for the CFG 4.5 used at
generation), reimplemented in a small script around the frozen model + LoRA, because MMAudio's runner is built for
multi-GPU training from scratch with its own data folders.

Settings (start values, changed only if the curves say so): AdamW, learning rate 1e-4 with short warm-up, batch 8,
bf16, ~6,000 steps (~8 epochs), validation loss every 500 steps, keep the best. ~2–3 h per run on the RTX 4080.

> **Learn — sanity check first.** Before the real run we train on only 8 clips for a few hundred steps. A working
> setup must almost memorise them (training loss drops far). If it doesn't, something is broken (wrong data,
> frozen adapters, learning rate) and a 3-hour run would be wasted.

## 5. Evaluation (same as CP9 / CP13, so rows are comparable)

On GAPS test (74 clips, real MIDI), GuiFiR-Vid (32) and magcil (23, both pseudo-GT), masked to frames with strips:
generate with seed 42, 25 steps, ± prompt → transcribe with the GAPS model (`30d`) → `30e` (note F1@50/100, frame F1,
timing F1, chroma, onset r) + **other-video control** + GuiFiR tab recall. Rows: MMAudio zero-shot, + LoRA (A),
+ LoRA text-only (B).

What would answer the question:
- **A > zero-shot and A > B on note F1, and A better with own video than another** → LoRA learned something about
  notes from the video.
- **A ≈ B > zero-shot** → it learned the guitar sound/music style only.
- GuiFiR and magcil (steel-string, electric) show whether it transfers beyond classical GAPS videos.

## 6. Steps (each ends with something to look at)

| step | what | you see |
|---|---|---|
| 16a | windows + sync + performer split + leakage check; feature extraction | counts, hours, split table |
| 16b | LoRA wrapper + 8-clip sanity run | loss curve going down, parameter count |
| 16c | runs A and B | training/validation curves, a few generated spectrograms vs real |
| 16d | generation + evaluation on 3 sets, table rows | rows per dataset, spectrograms |

Time: ~1 day of compute + checks; target ~Dec 5 (rule: no results after ~4 days → main experiments phase).

## Decisions (Luckey, 2026-11-26)

1. **Control run B: yes.** Without it a better note F1 could just mean "sounds more like a classical guitar";
   B (same training, video blanked) separates that from "learned the notes from the video".
2. **Rank 16** as the single setting for now. Rank 8 / 32 comparison only later, if time allows.
