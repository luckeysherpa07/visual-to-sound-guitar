"""CP16: LoRA fine-tune of MMAudio on GAPS train (plan: docs/notes/2026-11-26_cp16_mmaudio_lora_plan.md).

Step 16a (this part): training windows and their MMAudio features.

- ``make_windows`` / ``performer_split`` (any env): 8 s windows through each GAPS train video, a held-out validation
  part by performer.
- ``WindowFeatureExtractor`` (env ``mmaudio`` only): for the windows of one video, MMAudio's own conditioning
  features + audio latents, computed exactly like at generation time (``mmaudio.eval_utils.load_video`` transforms,
  ``FeaturesUtils`` encoders) and like MMAudio's training data (audio = the mp4's own audio track, mono, 44.1 kHz,
  353,280 samples = 8.01 s per window, VAE latent mean/std):
    clip   (64, 1024)   CLIP image features, 8 fps
    sync   (192, 768)   Synchformer features, 24 per s
    mean, std (345, 40) VAE latent distribution of the real audio (the training target is sampled from it)
  The text features of the prompt are the same for every window and stored once.

Video frames: each window is cut like the eval clips and read with MMAudio's ``load_video`` (see ``video_windows``).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

WINDOW_S = 8.0
CLIP_FPS, SYNC_FPS = 8.0, 25.0
SR = 44100
N_SAMPLES = 353_280  # mmaudio.model.sequence_config.CONFIG_44K.num_audio_frames
PROMPT = "solo classical guitar"  # the GAPS prompt of CP4/CP9


# ---------------------------------------------------------------- windows and split (any env)
def make_windows(video_s: float, length: float = WINDOW_S, audio_s: float = N_SAMPLES / SR) -> list[float]:
    """Window starts every ``length`` s that fit inside the video (frames and the 8.01 s of audio)."""
    starts, s = [], 0.0
    while s + max(length, audio_s) <= video_s - 0.05:
        starts.append(round(s, 3))
        s += length
    return starts


def performer_split(pieces: list[dict], val_share: float = 0.1, seed: int = 0, max_pieces: int = 3) -> set[str]:
    """Piece ids of the validation part: whole performers drawn at random until >= val_share of the pieces.

    Unknown/empty performers count as one performer per piece. Only performers with <= max_pieces pieces are drawn,
    so no single prolific performer dominates the validation part.
    """
    groups: dict[str, list[str]] = {}
    for p in pieces:
        name = (p["performer"] or "").strip()
        key = p["id"] if name in ("", "Unknown") else name
        groups.setdefault(key, []).append(p["id"])
    keys = sorted(k for k, v in groups.items() if len(v) <= max_pieces)
    random.Random(seed).shuffle(keys)
    val, target = set(), val_share * len(pieces)
    for k in keys:
        if len(val) >= target:
            break
        val.update(groups[k])
    return val


def read_windows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- features (env mmaudio)
class WindowFeatureExtractor:
    def __init__(self, weights_dir: Path | None = None, device: str = "cuda"):
        import torch
        from mmaudio.eval_utils import all_model_cfg
        from mmaudio.model.utils.features_utils import FeaturesUtils

        from guitar2midi.baselines.v2a_mmaudio import WEIGHTS_DIR, _relocate

        self.torch = torch
        cfg = _relocate(all_model_cfg["large_44k_v2"], weights_dir or WEIGHTS_DIR)
        cfg.download_if_needed()
        self.device = device
        # CLIP + Synchformer in bf16 as at generation time (v2a_mmaudio); the audio VAE encoder in fp32 (its mel
        # converter / STFT does not run in bf16)
        self.dtype = torch.bfloat16
        self.features = FeaturesUtils(tod_vae_ckpt=cfg.vae_path, synchformer_ckpt=cfg.synchformer_ckpt,
                                      enable_conditions=True, mode=cfg.mode, bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
                                      need_vae_encoder=True).to(device, torch.float32).eval()
        self.features.clip_model.to(self.dtype)
        self.features.synchformer.to(self.dtype)

    def text(self, prompt: str = PROMPT):
        with self.torch.inference_mode():
            return self.features.encode_text([prompt]).float().cpu()[0]  # (77, 1024)

    @staticmethod
    def read_audio(video: Path) -> np.ndarray:
        """Whole audio track of the mp4 as mono float32 at 44.1 kHz."""
        import av
        import torch
        import torchaudio.functional as AF

        with av.open(str(video)) as c:
            st = c.streams.audio[0]
            sr = st.rate
            planar = av.AudioResampler(format="fltp", layout=st.layout.name, rate=sr)  # float, (channels, n)
            chunks = [g.to_ndarray() for f in c.decode(st) for g in planar.resample(f)]
            chunks += [g.to_ndarray() for g in planar.resample(None)]
        x = np.concatenate(chunks, axis=1).astype(np.float32).mean(0)  # mono = channel mean, as MMAudio
        if sr != SR:
            x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
        return x

    def video_windows(self, video: Path, starts: list[float], tmp: Path, workers: int = 4) -> dict[float, dict]:
        """Features for the given windows of one video (windows with too few frames are skipped).

        Frames: every window is cut exactly like the eval clips (``clips.cut_video``: ffmpeg ``fps=25`` filter,
        x264 CRF 18) and read with MMAudio's own ``load_video``, so training sees the same input as evaluation
        (an own single-pass reader differed by ~1 frame in timing, and Synchformer features are sensitive to that:
        cosine 0.83). Cuts run in ``workers`` ffmpeg processes ahead of the GPU.
        Audio: the mp4's own audio track, ``N_SAMPLES`` from the window start.
        """
        from concurrent.futures import ThreadPoolExecutor

        from mmaudio.eval_utils import load_video

        from guitar2midi.baselines.clips import cut_video

        torch = self.torch
        audio = self.read_audio(video)
        tmp.mkdir(parents=True, exist_ok=True)

        def cut(s: float) -> Path:
            dst = tmp / f"{video.stem}_{int(round(s * 1000)):07d}.mp4"
            cut_video(video, dst, s, WINDOW_S + 0.2)  # 8.0 s cuts give load_video only 199 sync frames
            return dst

        res = {}
        with ThreadPoolExecutor(workers) as pool, torch.inference_mode():
            for s, clip_file in zip(starts, pool.map(cut, starts)):
                info = load_video(clip_file, WINDOW_S)
                clip_file.unlink()
                a0 = int(round(s * SR))
                wav = audio[a0:a0 + N_SAMPLES]
                if (info.clip_frames.shape[0] < CLIP_FPS * WINDOW_S or info.sync_frames.shape[0] < SYNC_FPS * WINDOW_S
                        or len(wav) < N_SAMPLES):
                    continue
                clip_f = self.features.encode_video_with_clip(info.clip_frames[None].to(self.device, self.dtype))[0]
                sync_f = self.features.encode_video_with_sync(info.sync_frames[None].to(self.device, self.dtype))[0]
                dist = self.features.encode_audio(torch.from_numpy(wav).to(self.device, torch.float32)[None])
                res[s] = {"clip": clip_f.float().cpu().numpy(), "sync": sync_f.float().cpu().numpy(),
                          "mean": dist.mean[0].transpose(0, 1).float().cpu().numpy(),
                          "std": dist.std[0].transpose(0, 1).float().cpu().numpy(),
                          "rms_db": float(20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9))}
        return res


# ---------------------------------------------------------------- training (env mmaudio; step 16b/16c)
# MMAudio's training settings (its config/train_config.yaml + base_config.yaml, runner.py):
LORA_TARGETS = ("attn.qkv", "linear1", "ffn.w1", "ffn.w2", "ffn.w3")  # attention in-projection, attention output,
# MLP; in all 21 blocks (latent/clip/text streams of the 7 joint blocks + 14 fused blocks). Not adaLN_modulation
# (the time/condition modulation) and not the input projections.
NULL_COND_P = 0.1          # null_condition_probability: video / text each replaced by "empty" 10 % of the time
T_MEAN, T_SCALE = 0.0, 1.0  # sampling.mean / scale: t = sigmoid(N(0, 1)) (log-normal sampling of the noise level)


def load_features(feat_dir: Path, pieces: list[str], per_piece: int | None = None) -> dict:
    """Stored windows of ``pieces`` as CPU tensors (clip/sync fp16, mean/std fp32), all or the first ``per_piece``
    windows of each piece, plus ``other``: for each window a fixed window of another piece (video control)."""
    import torch

    parts = {k: [] for k in ("clip", "sync", "mean", "std")}
    ids = []
    for piece in pieces:
        z = np.load(feat_dir / f"{piece}.npz")
        n = len(z["start_s"]) if per_piece is None else min(per_piece, len(z["start_s"]))
        for k in parts:
            parts[k].append(z[k][:n])
        ids += [f"{piece}_{int(round(s * 1000)):07d}" for s in z["start_s"][:n]]
    out = {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}
    out["ids"] = ids
    piece_of = np.array([i.rsplit("_", 1)[0] for i in ids])
    rng = np.random.default_rng(0)
    out["other"] = torch.tensor([rng.choice(np.flatnonzero(piece_of != p)) if (piece_of != p).any() else i
                                 for i, p in enumerate(piece_of)])
    return out


class LoRATrainer:
    """MMAudio large_44k_v2 (frozen, fp32 weights, bf16 autocast as MMAudio's own training) + LoRA adapters.

    ``video=False`` = control run B: the video features are always MMAudio's learned "empty video" features, so the
    adapters can only learn what the audio of guitar videos sounds like in general, not what this video shows.
    """

    def __init__(self, text_features: np.ndarray, rank: int = 16, alpha: float | None = None, video: bool = True,
                 weights_dir: Path | None = None, device: str = "cuda", seed: int = 0):
        import torch
        from mmaudio.eval_utils import all_model_cfg
        from mmaudio.model.flow_matching import FlowMatching
        from mmaudio.model.networks import get_my_mmaudio

        from guitar2midi.baselines import lora
        from guitar2midi.baselines.v2a_mmaudio import WEIGHTS_DIR, _relocate

        self.torch, self.device, self.video = torch, device, video
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        cfg = _relocate(all_model_cfg["large_44k_v2"], weights_dir or WEIGHTS_DIR)
        cfg.download_if_needed()
        self.net = get_my_mmaudio(cfg.model_name).to(device).eval()  # eval: no dropout in MMAudio anyway
        self.net.load_weights(torch.load(cfg.model_path, map_location=device, weights_only=True))
        self.n_total = sum(p.numel() for p in self.net.parameters())
        self.wrapped = lora.add_lora(self.net, rank, alpha, LORA_TARGETS, within=("joint_blocks", "fused_blocks"))
        self.params = lora.lora_parameters(self.net)
        self.n_lora = sum(p.numel() for p in self.params)
        self.fm = FlowMatching(min_sigma=0.0)
        self.text = torch.from_numpy(np.asarray(text_features)).float().to(device)
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.lora = lora

    def _batch(self, data: dict, idx, gen, dropout: bool, video_mode: str = "own"):
        """Conditions and noised latents for windows ``idx`` (MMAudio's train_fn / val_fn)."""
        torch, net, dev = self.torch, self.net, self.device
        bs = len(idx)
        vid = data["other"][idx] if video_mode == "other" else idx  # "other": a window of another piece (control)
        clip = data["clip"][vid].to(dev).float()
        sync = data["sync"][vid].to(dev).float()
        text = self.text[None].expand(bs, -1, -1).clone()
        mean, std = data["mean"][idx].to(dev), data["std"][idx].to(dev)
        x1 = mean + std * torch.empty_like(mean).normal_(generator=gen)
        x1 = net.normalize(x1)
        t = torch.sigmoid(torch.randn(bs, device=dev, generator=gen) * T_SCALE + T_MEAN)
        x0 = torch.empty_like(x1).normal_(generator=gen)
        xt = self.fm.get_conditional_flow(x0, x1, t)
        null_v = torch.rand(bs, device=dev, generator=gen) < NULL_COND_P
        null_t = torch.rand(bs, device=dev, generator=gen) < NULL_COND_P
        if not dropout:
            null_v[:], null_t[:] = False, False
        if not self.video or video_mode == "blank":
            null_v[:] = True
        clip[null_v] = net.empty_clip_feat
        sync[null_v] = net.empty_sync_feat
        text[null_t] = net.empty_string_feat
        return xt, clip, sync, text, t, x0, x1

    def loss(self, data: dict, idx, gen=None, dropout: bool = True, video_mode: str = "own"):
        """Per-window flow-matching loss (B,)."""
        torch = self.torch
        xt, clip, sync, text, t, x0, x1 = self._batch(data, idx, gen or self.rng, dropout, video_mode)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.net(xt, clip, sync, text, t)
        return self.fm.loss(pred.float(), x0, x1)

    def eval_loss(self, data: dict, batch: int = 16, seed: int = 1234, video_mode: str = "own") -> np.ndarray:
        """Per-window loss with fixed noise / noise levels (same draws every call, so curves are comparable) and no
        condition dropout. video_mode: own / other (a window of another piece) / blank (empty video)."""
        torch = self.torch
        gen = torch.Generator(device=self.device).manual_seed(seed)
        out = []
        with torch.no_grad():
            for i in range(0, len(data["ids"]), batch):
                idx = torch.arange(i, min(i + batch, len(data["ids"])))
                out.append(self.loss(data, idx, gen, dropout=False, video_mode=video_mode).cpu().numpy())
        return np.concatenate(out)

    def adapters(self) -> dict:
        return self.lora.lora_state_dict(self.net)
