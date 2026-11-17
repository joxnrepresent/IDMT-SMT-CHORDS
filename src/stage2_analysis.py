"""Generate the Stage 2 CQT comparison and per-instrument brightness table."""
import csv
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
LABELS_PATH = PROJECT_DIR / "data" / "processed" / "labels.csv"
CLIPS_PATH = PROJECT_DIR / "data" / "processed" / "clips_16k.npy"
RESULTS_DIR = PROJECT_DIR / "results"
SAMPLE_RATE = 16000


def read_labels():
    with LABELS_PATH.open(newline="") as labels_file:
        return list(csv.DictReader(labels_file))


def plot_cqt_comparison(labels, clips, instruments):
    selected_instruments = [instruments[index] for index in (0, 5, 12)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, instrument in zip(axes, selected_instruments):
        clip_index = next(
            index
            for index, row in enumerate(labels)
            if row["harte"] == "C:maj" and row["instrument"] == instrument
        )
        cqt = np.abs(
            librosa.cqt(
                clips[clip_index].astype(np.float32),
                sr=SAMPLE_RATE,
                hop_length=256,
                fmin=librosa.note_to_hz("C1"),
                n_bins=84,
            )
        )
        librosa.display.specshow(
            librosa.amplitude_to_db(cqt, ref=np.max),
            sr=SAMPLE_RATE,
            hop_length=256,
            fmin=librosa.note_to_hz("C1"),
            x_axis="time",
            y_axis="cqt_note",
            ax=ax,
            cmap="magma",
        )
        ax.set_title(f"C:maj on {instrument.replace('_', ' ')}")
    axes[0].set_ylabel("Note")
    fig.suptitle("C:maj Constant-Q Spectrograms Across Three Instruments")
    fig.tight_layout()
    output_path = RESULTS_DIR / "fig2_cqt.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def write_brightness_table(labels, clips):
    centroid_by_instrument = {}
    family_by_instrument = {}
    for index, row in enumerate(labels):
        instrument = row["instrument"]
        clip = clips[index].astype(np.float32)
        centroid = librosa.feature.spectral_centroid(
            y=clip,
            sr=SAMPLE_RATE,
        )[0].mean()
        centroid_by_instrument.setdefault(instrument, []).append(centroid)
        family_by_instrument[instrument] = row["family"]

    summaries = []
    for instrument, centroids in centroid_by_instrument.items():
        centroids = np.asarray(centroids)
        summaries.append({
            "instrument": instrument,
            "family": family_by_instrument[instrument],
            "median": np.median(centroids),
            "q25": np.percentile(centroids, 25),
            "q75": np.percentile(centroids, 75),
        })
    summaries.sort(key=lambda row: row["median"])

    output_path = RESULTS_DIR / "table1_brightness.csv"
    with output_path.open("w", newline="") as table_file:
        writer = csv.DictWriter(
            table_file,
            fieldnames=["instrument", "family", "median", "q25", "q75"],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                **row,
                "median": f"{row['median']:.0f}",
                "q25": f"{row['q25']:.0f}",
                "q75": f"{row['q75']:.0f}",
            })
    print(f"Saved {output_path}")
    return summaries


def main():
    labels = read_labels()
    clips = np.load(CLIPS_PATH, mmap_mode="r")
    if len(labels) != len(clips):
        raise ValueError(f"{len(labels)} labels do not match {len(clips)} clips")

    instruments = list(dict.fromkeys(row["instrument"] for row in labels))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_cqt_comparison(labels, clips, instruments)
    summaries = write_brightness_table(labels, clips)
    print(f"Calculated brightness for {len(summaries)} instruments")


if __name__ == "__main__":
    main()
