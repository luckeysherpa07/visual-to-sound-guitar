"""Spectrogram comparison of real vs generated audio (for checking V2A output by eye)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile

    sr, raw = wavfile.read(str(path))
    x = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        x /= np.iinfo(raw.dtype).max
    return (x.mean(1) if x.ndim == 2 else x), sr


def log_spectrogram(x: np.ndarray, sr: int, fmax: float = 4000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dB magnitude, 4096-sample window (enough pitch resolution for guitar notes), 10 ms hop."""
    from scipy.signal import spectrogram

    f, t, s = spectrogram(x, sr, nperseg=4096, noverlap=4096 - int(sr * 0.01), mode="magnitude")
    keep = f <= fmax
    return f[keep], t, 20 * np.log10(s[keep] + 1e-6)


def plot_spectrograms(audio: dict[str, Path], dst: Path, title: str = "", gt_notes: np.ndarray | None = None,
                      duration: float = 8.0, ref_label: str = "GT notes") -> None:
    """One row per audio file (log-frequency axis up to 4 kHz). Optional GT notes (onset, offset, pitch)
    are drawn as white lines at their fundamental frequency on every row, to compare pitch and timing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(audio), 1, figsize=(12, 2.6 * len(audio)), sharex=True, squeeze=False)
    for ax, (label, path) in zip(axes[:, 0], audio.items()):
        x, sr = load_wav_mono(path)
        f, t, s = log_spectrogram(x, sr)
        ax.pcolormesh(t, f, s, vmin=s.max() - 70, vmax=s.max(), cmap="magma", shading="auto")
        ax.set_yscale("log")
        ax.set_ylim(70, 4000)
        ax.set_ylabel("Hz")
        ax.set_title(label, loc="left", fontsize=10)
        if gt_notes is not None:
            for on, off, pitch in gt_notes:
                hz = 440.0 * 2 ** ((pitch - 69) / 12)
                ax.plot([on, off], [hz, hz], color="white", lw=0.8, alpha=0.7)
    axes[-1, 0].set_xlabel("time in clip (s)")
    axes[-1, 0].set_xlim(0, duration)
    fig.suptitle(title + (f"   (white lines = {ref_label})" if gt_notes is not None else ""))
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=100)
    plt.close(fig)
