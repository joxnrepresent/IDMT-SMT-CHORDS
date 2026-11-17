"""Compare chord-aligned E:maj chroma features across all instrument recordings."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.ticker import FixedLocator, FixedFormatter

from plot_chroma import (
    PITCH_CLASSES,
    compute_chroma_for_interval,
    read_annotations,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_DIR / "data" / "audio"
OUTPUT_PATH = PROJECT_DIR / "plots" / "chroma_Emaj_all_instruments.png"
FAMILIES = ("guitar", "non_guitar")


def is_e_major(label):
    root, chord = label.split(":", 1)
    quality = chord.split("/", 1)[0]
    return root == "E" and quality == "maj"


def main():
    recordings = []
    for family in FAMILIES:
        family_dir = AUDIO_DIR / family
        annotations = read_annotations(family_dir / f"{family}_annotation.lab")
        e_major_chords = [row for row in annotations if is_e_major(row[2])]
        for audio_path in sorted(family_dir.glob("*.wav")):
            recordings.append((audio_path, e_major_chords, annotations[-1][1]))

    columns = 4
    rows = (len(recordings) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(22, 14),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    image = None

    for ax, (audio_path, e_major_chords, annotation_end) in zip(axes, recordings):
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        duration = len(audio) / sample_rate
        if not np.isclose(annotation_end, duration):
            family = audio_path.parent.name
            raise ValueError(
                f"{audio_path.name}: {family} annotations end at "
                f"{annotation_end} s but audio is {duration} s"
            )

        chroma = np.asarray([
            compute_chroma_for_interval(audio, sample_rate, start, end)
            for start, end, _ in e_major_chords
        ]).T
        image = ax.imshow(
            chroma,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(0, len(e_major_chords), -0.5, len(PITCH_CLASSES) - 0.5),
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        ax.set_title(audio_path.stem.replace("_", " "), fontsize=8)
        ax.set_yticks(range(len(PITCH_CLASSES)), PITCH_CLASSES, fontsize=7)
        centers = np.arange(len(e_major_chords)) + 0.5
        start_labels = [
            f"{int(start)}"
            for start, _, _ in e_major_chords
        ]
        ax.xaxis.set_major_locator(FixedLocator(centers))
        ax.xaxis.set_major_formatter(FixedFormatter(start_labels))
        ax.tick_params(axis="x", labelsize=7)

    for ax in axes[len(recordings):]:
        ax.set_visible(False)

    fig.suptitle(
        "E:maj Chroma Features by Instrument\n"
        "(includes E:maj inversions; excludes E:maj7)",
        fontsize=15,
    )
    fig.supxlabel("E:maj occurrence start time in recording (seconds)")
    fig.supylabel("Pitch class")
    colorbar = fig.colorbar(image, ax=axes[:len(recordings)].tolist(), shrink=0.82)
    colorbar.set_label("Mean normalized chroma")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(
        f"Saved {OUTPUT_PATH} for {len(recordings)} instruments "
        f"({len(recordings[0][1])} guitar and "
        f"{len(recordings[-1][1])} non-guitar E:maj instances each)"
    )


if __name__ == "__main__":
    main()
