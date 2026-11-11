"""Step 0: inspect the IDMT-SMT-Chords dataset (no modelling code).

Run from anywhere:  python src/inspect_data.py
Prints a table of every file and saves results/step0_first_3s.png.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDERS = ["guitar", "non_guitar"]
RESULTS_DIR = PROJECT_DIR / "results"
ONSET_THRESHOLD = 0.01  # a sample counts as "sound" if |x| > 1% of full scale


def list_all_files():
    """Print every file in the dataset folders with its size in MB."""
    print("=== All files ===")
    for folder in DATA_FOLDERS:
        for path in sorted((PROJECT_DIR / folder).iterdir()):
            size_mb = path.stat().st_size / 1e6
            print(f"{folder}/{path.name:55s} {size_mb:8.2f} MB")
    print()


def read_lab(path):
    """Read a .lab annotation file into a DataFrame with columns start, end, label."""
    table = pd.read_csv(path, sep="\t", header=None, names=["start", "end", "label"])
    table["label"] = table["label"].str.strip()  # the file has Windows line endings (\r\n)
    return table


def check_lab_grid(table):
    """Return True if row i covers exactly [2*i, 2*i + 2] seconds."""
    expected_start = 2.0 * np.arange(len(table))
    starts_ok = np.allclose(table["start"], expected_start)
    ends_ok = np.allclose(table["end"], expected_start + 2.0)
    return starts_ok and ends_ok


def first_sound_time(signal, sr):
    """Time in seconds of the first sample louder than ONSET_THRESHOLD (None if silent)."""
    loud = np.flatnonzero(np.abs(signal) > ONSET_THRESHOLD)
    if len(loud) == 0:
        return None
    return loud[0] / sr


def inspect_wav(path, lab_table):
    """Collect duration, sample rate, channels, format and onset checks for one WAV."""
    info = sf.info(str(path))
    duration = info.frames / info.samplerate

    # First chord: is there a count-in before 0.0 s?
    start_audio, sr = sf.read(str(path), frames=3 * info.samplerate)
    first_onset = first_sound_time(start_audio, sr)

    # Last chord: does it still start where the .lab file says (no drift)?
    last_start = lab_table["start"].iloc[-1]
    last_audio, _ = sf.read(str(path), start=int(last_start * sr))
    last_onset = first_sound_time(last_audio, sr)

    return {
        "file": f"{path.parent.name}/{path.name}",
        "duration_s": duration,
        "sr": info.samplerate,
        "channels": info.channels,
        "format": info.subtype,
        "first_onset_s": first_onset,
        "last_chord_onset_after_start_s": last_onset,
    }

def plot_first_3_seconds(paths):
    """Plot the first 3 s of each given WAV, with dashed lines at the 2 s chord boundaries."""
    fig, axes = plt.subplots(len(paths), 1, figsize=(10, 2.2 * len(paths)), sharex=True)
    for ax, path in zip(axes, paths):
        signal, sr = sf.read(str(path), frames=3 * sf.info(str(path)).samplerate)
        time = np.arange(len(signal)) / sr
        ax.plot(time, signal, linewidth=0.5)
        for boundary in [0.0, 2.0]:
            ax.axvline(boundary, color="red", linestyle="--", linewidth=1)
        ax.set_title(path.name, fontsize=9)
        ax.set_ylabel("amplitude")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out_path = RESULTS_DIR / "step0_first_3s.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


def main():
    list_all_files()

    rows = []
    for folder in DATA_FOLDERS:
        lab_path = PROJECT_DIR / folder / f"{folder}_annotation.lab"
        lab_table = read_lab(lab_path)
        print(f"{lab_path.name}: {len(lab_table)} chords, "
              f"last end = {lab_table['end'].iloc[-1]} s, "
              f"exact 2 s grid from 0.0 s: {check_lab_grid(lab_table)}")

        for wav_path in sorted((PROJECT_DIR / folder).glob("*.wav")):
            row = inspect_wav(wav_path, lab_table)
            row["n_labels"] = len(lab_table)
            row["duration_matches_lab"] = np.isclose(row["duration_s"], lab_table["end"].iloc[-1])
            rows.append(row)
    print()

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nTotal chords (labels x files): {table['n_labels'].sum()}")

    # Plot one guitar file and one non-guitar file
    example_paths = [
        sorted((PROJECT_DIR / "guitar").glob("*.wav"))[0],
        sorted((PROJECT_DIR / "non_guitar").glob("*.wav"))[0],
    ]
    plot_first_3_seconds(example_paths)


if __name__ == "__main__":
    main()
