"""Compare per-clip spectral centroid distributions across instruments."""
import csv
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
LABELS_PATH = PROJECT_DIR / "data" / "processed" / "labels.csv"
CLIPS_PATH = PROJECT_DIR / "data" / "processed" / "clips_16k.npy"
OUTPUT_PATH = Path(__file__).resolve().parent / "spectral_centroid_by_instrument.png"
SAMPLE_RATE = 16000


def main():
    with LABELS_PATH.open(newline="") as labels_file:
        labels = list(csv.DictReader(labels_file))
    clips = np.load(CLIPS_PATH, mmap_mode="r")
    if len(labels) != len(clips):
        raise ValueError(f"{len(labels)} labels do not match {len(clips)} clips")

    instrument_names = list(dict.fromkeys(row["instrument"] for row in labels))
    centroids = {instrument: [] for instrument in instrument_names}
    families = {}
    for index, row in enumerate(labels):
        centroid_hz = librosa.feature.spectral_centroid(
            y=clips[index].astype(np.float32),
            sr=SAMPLE_RATE,
        )[0].mean()
        if not np.isfinite(centroid_hz) or centroid_hz <= 0:
            raise ValueError(
                f"Invalid spectral centroid for clip {row['clip_id']}: {centroid_hz}"
            )
        centroids[row["instrument"]].append(centroid_hz)
        families[row["instrument"]] = row["family"]

    fig, ax = plt.subplots(figsize=(16, 8))
    boxes = ax.boxplot(
        [centroids[instrument] for instrument in instrument_names],
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
    )
    family_colors = {"guitar": "#4C78A8", "non_guitar": "#F58518"}
    for instrument, box in zip(instrument_names, boxes["boxes"]):
        box.set_facecolor(family_colors[families[instrument]])
        box.set_alpha(0.7)

    ax.set_yscale("log")
    ax.set_xticks(
        np.arange(1, len(instrument_names) + 1),
        [instrument.replace("_", " ") for instrument in instrument_names],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Spectral centroid (Hz, log scale)")
    ax.set_xlabel("Instrument")
    ax.set_title("Spectral Centroid Distribution by Instrument")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=family_colors["guitar"], label="Guitar"),
            plt.Rectangle((0, 0), 1, 1, color=family_colors["non_guitar"], label="Non-guitar"),
        ],
        title="Instrument family",
        loc="upper left",
    )
    fig.text(
        0.5,
        0.015,
        "Each box summarizes the mean frame-wise spectral centroid of labeled 2-second clips; "
        "diamonds mark the instrument's mean across clips.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH} for {len(instrument_names)} instruments")


if __name__ == "__main__":
    main()
