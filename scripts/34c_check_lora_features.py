"""CP16 step 16a check: is the stored training target the real audio? (env mmaudio)

    PYTHONNOUSERSITE=1 python scripts/34c_check_lora_features.py          # env mmaudio: decode, -> check.npz
    python scripts/34c_check_lora_features.py --plot                      # env guitar2midi: check_sheet.png

For a few windows (loud / median / quiet training window + one validation window): the real audio of the window
(mp4 track) next to the stored VAE latent mean decoded back to sound (VAE decoder + BigVGAN vocoder), with the
GAPS notes (video time, lag ignored) on top, plus the correlation of the two loudness envelopes. Shows what the
LoRA is trained to produce.

Writes outputs/runs/2026-11-26_cp16_features_check/{check.npz, envelope_corr.txt, check_sheet.png}.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from guitar2midi.baselines import mmaudio_lora as L

D = Path("data/processed/mmaudio_lora")
GAPS = Path("data/raw/gaps")
OUT = Path("outputs/runs/2026-11-26_cp16_features_check")


def spec_db(x: np.ndarray) -> np.ndarray:
    s = torch.stft(torch.from_numpy(x), 2048, 512, window=torch.hann_window(2048), return_complex=True).abs()
    return 20 * np.log10(s.numpy()[:400] + 1e-5)  # up to ~8.6 kHz


def envelope(x: np.ndarray, hop: int = 441) -> np.ndarray:
    n = len(x) // hop
    return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))


def compute() -> None:
    split = {r["piece"]: r["split"] for r in csv.DictReader(open(D / "split.csv"))}
    items = []  # (piece, start, rms, split)
    for f in sorted((D / "features").glob("*.npz")):
        z = np.load(f)
        items += [(f.stem, float(s), float(r), split[f.stem]) for s, r in zip(z["start_s"], z["rms_db"])]
    tr = sorted((i for i in items if i[3] == "train"), key=lambda i: i[2])
    va = sorted((i for i in items if i[3] == "val"), key=lambda i: i[2])
    pick = [("train, loud", tr[int(0.95 * len(tr))]), ("train, median", tr[len(tr) // 2]),
            ("train, quiet", tr[int(0.05 * len(tr))]), ("val, median", va[len(va) // 2])]

    ex = L.WindowFeatureExtractor()
    OUT.mkdir(parents=True, exist_ok=True)
    out, lines = {}, []
    for row, (label, (piece, s, rms, _)) in enumerate(pick):
        z = np.load(D / "features" / f"{piece}.npz")
        k = int(np.where(np.isclose(z["start_s"], s))[0][0])
        audio = ex.read_audio(GAPS / "videos" / f"{piece}.mp4")
        real = audio[int(round(s * L.SR)):int(round(s * L.SR)) + L.N_SAMPLES]
        with torch.inference_mode():
            lat = torch.from_numpy(z["mean"][k][None]).to(ex.device, torch.float32)
            dec = ex.features.vocode(ex.features.decode(lat)).float().cpu().numpy().reshape(-1)[:L.N_SAMPLES]
        e1, e2 = envelope(real), envelope(dec)
        n = min(len(e1), len(e2))
        r = float(np.corrcoef(e1[:n], e2[:n])[0, 1])
        lines.append(f"{label:14s} {piece} {s:7.1f} s  rms {rms:6.1f} dB  envelope r {r:.3f}")
        out[f"real{row}"], out[f"dec{row}"] = spec_db(real), spec_db(dec)
        out[f"meta{row}"] = np.array([label, piece, str(s), f"{rms:.1f}", f"{r:.3f}"])
    np.savez(OUT / "check.npz", **out)
    (OUT / "envelope_corr.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(OUT / "check.npz")
    rows = len([k for k in z.files if k.startswith("meta")])
    fig, ax = plt.subplots(rows, 2, figsize=(14, 3 * rows), sharey=True)
    for row in range(rows):
        label, piece, s, rms, r = z[f"meta{row}"]
        s = float(s)
        midi = pretty_midi.PrettyMIDI(str(GAPS / "midi" / f"{piece}.mid"))
        notes = [(n.start - s, n.end - s, n.pitch) for i in midi.instruments for n in i.notes
                 if s <= n.start < s + L.WINDOW_S]
        for col, (key, name) in enumerate([("real", "real audio (mp4)"), ("dec", "stored target, decoded")]):
            a = ax[row, col]
            a.imshow(z[f"{key}{row}"], origin="lower", aspect="auto", cmap="magma", vmin=-60, vmax=40,
                     extent=[0, L.N_SAMPLES / L.SR, 0, 400 * L.SR / 2048])
            for t0, t1, p in notes:
                f0 = 440 * 2 ** ((p - 69) / 12)
                a.plot([t0, min(t1, L.WINDOW_S)], [f0, f0], color="cyan", lw=1.2, alpha=0.8)
            a.set_ylim(0, 2000)
            a.set_title(f"{label}: {piece} @ {s:.0f} s ({rms} dB) - {name}" + (f", envelope r {r}" if col else ""),
                        fontsize=9)
        ax[row, 0].set_ylabel("Hz (cyan = GAPS notes)")
    for a in ax[-1]:
        a.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(OUT / "check_sheet.png", dpi=110)
    print(OUT / "check_sheet.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot", action="store_true")
    plot() if p.parse_args().plot else compute()
