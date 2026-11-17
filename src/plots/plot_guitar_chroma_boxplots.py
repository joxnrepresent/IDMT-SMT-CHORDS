"""Plot per-instrument distributions of guitar chroma means and variances."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from plot_chroma import (
    PITCH_CLASSES,
    chroma_frames_for_interval,
    read_annotations,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
GUITAR_DIR = PROJECT_DIR / "data" / "audio" / "guitar"
OUTPUT_DIR = PROJECT_DIR / "plots"
CHUNK_LENGTH_S = 30


def summarize_recording(audio_path, annotations):
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    duration = len(audio) / sample_rate
    if not np.isclose(annotations[-1][1], duration):
        raise ValueError(
            f"{audio_path.name}: annotations end at {annotations[-1][1]} s "
            f"but audio is {duration} s"
        )

    chroma_chunks = []
    for start in np.arange(0, duration, CHUNK_LENGTH_S):
        end = min(start + CHUNK_LENGTH_S, duration)
        chroma_chunks.append(
            chroma_frames_for_interval(audio, sample_rate, start, end)
        )
    chroma = np.concatenate(chroma_chunks, axis=1)
    return chroma.mean(axis=1), chroma.var(axis=1)


def save_boxplot(values_by_instrument, instruments, title, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(13, 7))
    boxes = ax.boxplot(
        values_by_instrument,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    colors = plt.get_cmap("tab10").colors
    for index, box in enumerate(boxes["boxes"]):
        box.set_facecolor(colors[index % len(colors)])
        box.set_alpha(0.65)

    ax.set_xticks(
        np.arange(1, len(instruments) + 1),
        instruments,
        rotation=25,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Guitar instrument")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.5,
        0.015,
        "Each box summarizes the 12 pitch-class statistics for that recording; "
        "diamond = mean across pitch classes.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    annotations = read_annotations(GUITAR_DIR / "guitar_annotation.lab")
    audio_paths = sorted(GUITAR_DIR.glob("*.wav"))
    instruments = []
    means = []
    variances = []

    for audio_path in audio_paths:
        mean_by_pitch, variance_by_pitch = summarize_recording(audio_path, annotations)
        instruments.append(audio_path.stem.replace("_", " "))
        means.append(mean_by_pitch)
        variances.append(variance_by_pitch)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_path = OUTPUT_DIR / "chroma_stft_mean_guitar_boxplot.png"
    variance_path = OUTPUT_DIR / "chroma_stft_variance_guitar_boxplot.png"
    save_boxplot(
        means,
        instruments,
        "Chroma STFT Mean by Guitar Instrument",
        "Mean normalized chroma",
        mean_path,
    )
    save_boxplot(
        variances,
        instruments,
        "Chroma STFT Variance by Guitar Instrument",
        "Variance of normalized chroma",
        variance_path,
    )
    print(f"Saved {mean_path}")
    print(f"Saved {variance_path}")
    print(
        f"Compared {len(instruments)} instruments; each box summarizes "
        f"{len(PITCH_CLASSES)} pitch-class statistics."
    )


if __name__ == "__main__":
    main()
