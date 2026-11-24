"""Compare root-position E:maj chroma features across guitar recordings."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from plot_chroma import (
    PITCH_CLASSES,
    compute_chroma_for_interval,
    read_annotations,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
GUITAR_DIR = PROJECT_DIR / "data" / "audio" / "guitar"
OUTPUT_PATH = Path(__file__).resolve().parent / "chroma_Emaj_root_position_by_guitar.png"


def main():
    annotations = read_annotations(GUITAR_DIR / "guitar_annotation.lab")
    e_major_chords = [row for row in annotations if row[2] == "E:maj"]
    if not e_major_chords:
        raise ValueError("No root-position E:maj intervals found in guitar annotations")

    instrument_names = []
    chroma_profiles = []
    for audio_path in sorted(GUITAR_DIR.glob("*.wav")):
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        duration = len(audio) / sample_rate
        if not np.isclose(annotations[-1][1], duration):
            raise ValueError(
                f"{audio_path.name}: annotations end at {annotations[-1][1]} s "
                f"but audio is {duration} s"
            )

        instance_chroma = np.asarray([
            compute_chroma_for_interval(audio, sample_rate, start, end)
            for start, end, _ in e_major_chords
        ])
        chroma_profiles.append(instance_chroma.mean(axis=0))
        instrument_names.append(audio_path.stem.replace("_", " "))

    profiles = np.asarray(chroma_profiles)
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(
        profiles,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(range(len(PITCH_CLASSES)), PITCH_CLASSES)
    ax.set_yticks(range(len(instrument_names)), instrument_names)
    ax.set_xlabel("Pitch class")
    ax.set_ylabel("Guitar instrument")
    ax.set_title("Root-Position E:maj Chroma by Guitar Instrument")

    for row in range(profiles.shape[0]):
        for column in range(profiles.shape[1]):
            value = profiles[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black" if value >= 0.55 else "white",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Mean normalized chroma")
    fig.text(
        0.5,
        0.015,
        f"Each cell averages the stable middle of {len(e_major_chords)} "
        "root-position E:maj intervals per instrument; guitar inversion "
        "intervals are not annotated in this dataset.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.97, 0.95))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(
        f"Saved {OUTPUT_PATH} for {len(instrument_names)} guitar instruments "
        f"and {len(e_major_chords)} root-position E:maj intervals each"
    )


if __name__ == "__main__":
    main()
