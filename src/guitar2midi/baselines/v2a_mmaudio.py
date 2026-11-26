"""Video-to-audio with MMAudio (Cheng et al., CVPR 2025). Runs in the ``mmaudio`` env (envs/mmaudio.yml).

Follows MMAudio's demo.py (large_44k_v2, 25 Euler steps, CFG 4.5, bf16) but loads the model once
for a whole folder and stores weights in ~/.cache/mmaudio instead of ./weights.

Input handling inside MMAudio (relevant for guitar video): CLIP sees frames at 8 fps squashed to
384x384; Synchformer (audio-visual sync) sees 25 fps frames resized to 224 on the short side and
CENTER-CROPPED to 224x224, so on wide shots part of the guitar may be outside the sync crop.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

WEIGHTS_DIR = Path.home() / ".cache" / "mmaudio"


def _relocate(cfg, root: Path):
    """Copy of a ModelConfig with every relative weight path moved under ``root``."""
    changes = {f.name: root / getattr(cfg, f.name) for f in dataclasses.fields(cfg)
               if isinstance(getattr(cfg, f.name), Path)}
    return dataclasses.replace(cfg, **changes)


class MMAudioGenerator:
    def __init__(self, variant: str = "large_44k_v2", num_steps: int = 25, cfg_strength: float = 4.5,
                 full_precision: bool = False, weights_dir: Path = WEIGHTS_DIR, lora: Path | None = None,
                 no_video: bool = False):
        """lora: adapters_*.pt of a CP16 run (34d; its config.json next to it gives rank/alpha), merged into the
        weights before casting to bf16. no_video: generate with MMAudio's empty video features (CP16 run B)."""
        import torch
        from mmaudio.eval_utils import all_model_cfg
        from mmaudio.model.flow_matching import FlowMatching
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.cfg = _relocate(all_model_cfg[variant], weights_dir)
        self.cfg.download_if_needed()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if full_precision else torch.bfloat16
        self.net = get_my_mmaudio(self.cfg.model_name).to(self.device).eval()
        self.net.load_weights(torch.load(self.cfg.model_path, map_location=self.device, weights_only=True))
        if lora is not None:
            import json

            from guitar2midi.baselines import lora as lora_lib
            from guitar2midi.baselines.mmaudio_lora import LORA_TARGETS

            run_cfg = json.loads((Path(lora).parent / "config.json").read_text())
            lora_lib.add_lora(self.net, run_cfg["rank"], run_cfg["alpha"], LORA_TARGETS,
                              within=("joint_blocks", "fused_blocks"))
            lora_lib.load_lora(self.net, torch.load(lora, map_location=self.device, weights_only=True))
            lora_lib.merge_lora(self.net)
        self.net = self.net.to(dtype)
        self.no_video = no_video
        self.fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)
        self.features = FeaturesUtils(tod_vae_ckpt=self.cfg.vae_path, synchformer_ckpt=self.cfg.synchformer_ckpt,
                                      enable_conditions=True, mode=self.cfg.mode,
                                      bigvgan_vocoder_ckpt=self.cfg.bigvgan_16k_path,
                                      need_vae_encoder=False).to(self.device, dtype).eval()
        self.cfg_strength = cfg_strength
        self.sampling_rate = self.cfg.seq_cfg.sampling_rate

    def generate(self, video: Path, prompt: str = "", negative_prompt: str = "", duration: float = 8.0,
                 seed: int = 42, mask_away_clip: bool = False):
        """Returns (audio as float32 numpy (channels, samples), sampling rate, MMAudio VideoInfo)."""
        import torch
        from mmaudio.eval_utils import generate, load_video

        with torch.inference_mode():
            info = load_video(Path(video), duration)
            clip = None if (mask_away_clip or self.no_video) else info.clip_frames.unsqueeze(0)
            sync = None if self.no_video else info.sync_frames.unsqueeze(0)
            seq = self.cfg.seq_cfg
            seq.duration = info.duration_sec
            self.net.update_seq_lengths(seq.latent_seq_len, seq.clip_seq_len, seq.sync_seq_len)
            rng = torch.Generator(device=self.device)
            rng.manual_seed(seed)
            audio = generate(clip, sync, [prompt], negative_text=[negative_prompt], feature_utils=self.features,
                             net=self.net, fm=self.fm, rng=rng, cfg_strength=self.cfg_strength)
        return audio.float().cpu()[0].numpy(), self.sampling_rate, info

    def generate_dir(self, video_dir: Path, out_dir: Path, prompt: str = "", seed: int = 42,
                     duration: float = 8.0, overwrite: bool = False) -> list[Path]:
        """Generate ``out_dir/<clip>.wav`` for every ``*.mp4`` in ``video_dir`` (same seed for every clip)."""
        import soundfile as sf

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [v for v in sorted(Path(video_dir).glob("*.mp4")) if overwrite or not (out_dir / f"{v.stem}.wav").exists()]
        written = []
        for i, v in enumerate(todo, 1):
            audio, sr, _ = self.generate(v, prompt=prompt, seed=seed, duration=duration)
            dst = out_dir / f"{v.stem}.wav"
            sf.write(dst, audio.T, sr)
            written.append(dst)
            print(f"  [{i}/{len(todo)}] {v.name} -> {dst}", flush=True)
        return written


def mux(video: Path, audio: Path, dst: Path) -> None:
    """Silent clip + generated audio -> mp4 for listening (ffmpeg, video stream copied)."""
    import subprocess

    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio), "-map", "0:v",
                    "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)], check=True)
