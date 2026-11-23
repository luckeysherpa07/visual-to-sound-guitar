"""Audio -> MIDI with the GAPS benchmark guitar model (runs in the ``amt`` conda env, see envs/amt.yml).

Model: Kong et al. high-resolution CRNN (onset/offset/frame/velocity regression) trained on GAPS by
Riley et al. (ISMIR 2024), via github.com/xavriley/hf_midi_transcription. Input is resampled to 16 kHz
mono and processed in 10 s segments; an 8 s clip is zero-padded.
"""

from __future__ import annotations

from pathlib import Path

HF_REPO = "xavriley/midi-transcription-models"
CHECKPOINTS = {
    "gaps": "guitar-gaps.pth",  # package default for instrument="guitar"
    "gaps_paper": "guitar-gaps-paper-version-12200_iterations.pth",
}


def load_model(checkpoint: str = "gaps_paper", device: str = "auto", batch_size: int = 8):
    from huggingface_hub import hf_hub_download
    from hf_midi_transcription import MidiTranscriptionModel

    path = hf_hub_download(HF_REPO, CHECKPOINTS.get(checkpoint, checkpoint))
    return MidiTranscriptionModel(device=device, instrument="guitar", checkpoint_path=path, batch_size=batch_size)


def transcribe_dir(audio_dir: Path, out_dir: Path, checkpoint: str = "gaps_paper", pattern: str = "*.wav",
                   overwrite: bool = False) -> list[Path]:
    """Transcribe every audio file in ``audio_dir`` to ``out_dir/<stem>.mid``. Returns the written files."""
    files = sorted(Path(audio_dir).glob(pattern))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if overwrite or not (out_dir / f"{f.stem}.mid").exists()]
    if not todo:
        return []
    model = load_model(checkpoint)
    written = []
    for i, f in enumerate(todo, 1):
        dst = out_dir / f"{f.stem}.mid"
        model.transcribe(f, dst)
        written.append(dst)
        print(f"  [{i}/{len(todo)}] {f.name} -> {dst}", flush=True)
    return written
