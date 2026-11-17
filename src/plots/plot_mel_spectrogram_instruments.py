"""Compare all instrument timbres using the first annotated E:maj chord."""
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from plot_chroma import read_annotations

PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_DIR / "data" / "audio"
OUTPUT_PATH = Path(__file__).resolve().parent / "mel_spectrogram_all_instruments.png"
FAMILIES = ("guitar", "non_guitar")
N_MELS = 128
N_FFT = 4096
HOP_LENGTH = 512
FMIN = 20
FMAX = 16000
STABLE_MARGIN_S = 0.15


def main():
    recordings = []
    for family in FAMILIES:
        family_dir = AUDIO_DIR / family
        annotations = read_annotations(family_dir / f"{family}_annotation.lab")
        e_major = next(
            (row for row in annotations if row[2] == "E:maj"),
            None,
        )
        if e_major is None:
            raise ValueError(f"No root-position E:maj annotation in {family_dir}")
        for audio_path in sorted(family_dir.glob("*.wav")):
            recordings.append((audio_path, annotations, e_major))

    spectra = []
    for audio_path, annotations, (start, end, _) in recordings:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if not np.isclose(annotations[-1][1], len(audio) / sample_rate):
            raise ValueError(f"{audio_path.name}: audio and annotations do not align")

        stable_start = start + STABLE_MARGIN_S
        stable_end = end - STABLE_MARGIN_S
        segment = audio[
            round(stable_start * sample_rate):round(stable_end * sample_rate)
        ]
        fmax = min(FMAX, sample_rate / 2)
        mel_power = librosa.feature.melspectrogram(
            y=segment,
            sr=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=fmax,
            power=2.0,
        )
        spectra.append(
            (
                audio_path,
                start,
                end,
                sample_rate,
                fmax,
                librosa.power_to_db(mel_power, ref=np.max, top_db=70),
            )
        )

    columns = 4
    rows = (len(spectra) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(20, 13),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    image = None
    for ax, (audio_path, start, end, sample_rate, fmax, mel_db) in zip(axes, spectra):
        image = librosa.display.specshow(
            mel_db,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            fmin=FMIN,
            fmax=fmax,
            cmap="magma",
            vmin=-70,
            vmax=0,
            ax=ax,
        )
        ax.set_title(
            f"{audio_path.stem.replace('_', ' ')}\nE:maj ({start:g}-{end:g} s)",
            fontsize=8,
        )

    for ax in axes[len(spectra):]:
        ax.set_visible(False)

    fig.suptitle(
        "Mel-Spectrogram Comparison Across Instruments\n"
        "First root-position E:maj chord; each panel normalized to its own peak",
        fontsize=15,
    )
    colorbar = fig.colorbar(image, ax=axes[:len(spectra)].tolist(), shrink=0.82)
    colorbar.set_label("Magnitude (dB relative to each chord segment's peak)")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(f"Saved {OUTPUT_PATH} ({len(spectra)} instruments)")


if __name__ == "__main__":
    main()
