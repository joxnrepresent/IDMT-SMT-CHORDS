"""Step 1: build the label table (one row per 2 s chord clip).

Reads guitar/guitar_annotation.lab and non_guitar/non_guitar_annotation.lab.
Every WAV in a folder plays the same chord sequence, so each WAV gets a copy
of its folder's labels.

Run:  python src/labels.py
Writes data/processed/labels.csv
"""
from pathlib import Path

import pandas as pd

SEED = 42
PROJECT_DIR = Path(__file__).resolve().parent.parent
FAMILIES = ["guitar", "non_guitar"]
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "labels.csv"

# Root names used in the .lab files (sharps only). The index is the class: C=0 ... B=11.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Dataset shorthand -> our 7 quality class names
QUALITY_NAMES = {
    "maj": "maj",
    "min": "min",
    "maj7": "maj7",
    "min7": "min7",
    "7": "dom7",
    "hdim7": "m7b5",
    "5": "power",
}

# Semitones above the root for the chord's own 3rd, 5th and 7th.
# The dataset writes e.g. "C:min/3" meaning "the minor chord's own third" (Eb).
CHORD_DEGREES = {
    "maj":   {"3": 4, "5": 7},
    "min":   {"3": 3, "5": 7},
    "maj7":  {"3": 4, "5": 7, "7": 11},
    "min7":  {"3": 3, "5": 7, "7": 10},
    "dom7":  {"3": 4, "5": 7, "7": 10},
    "m7b5":  {"3": 3, "5": 6, "7": 10},
    "power": {"5": 7},
}

# Strict Harte names for those intervals (what mir_eval expects after the "/")
HARTE_INTERVAL_NAMES = {3: "b3", 4: "3", 6: "b5", 7: "5", 10: "b7", 11: "7"}


def read_lab(path):
    """Read a .lab file into a DataFrame with columns start_s, end_s, label."""
    table = pd.read_csv(path, sep="\t", header=None, names=["start_s", "end_s", "label"])
    table["label"] = table["label"].str.strip()  # remove the Windows '\r' at line ends
    return table


def parse_label(label):
    """Split a dataset label like 'C#:min7/3' into (root, quality, bass, harte).

    root and bass are pitch classes 0-11, quality is one of our 7 class names,
    harte is the strict Harte string (e.g. 'C#:min7/b3').
    """
    root_name, rest = label.split(":")

    # Separate the bass part, if there is one ("min7/3" -> "min7", "3")
    if "/" in rest:
        shorthand, bass_degree = rest.split("/")
    else:
        shorthand, bass_degree = rest, None

    root = NOTE_NAMES.index(root_name)
    quality = QUALITY_NAMES[shorthand]

    if bass_degree is None:
        bass = root
        harte = f"{root_name}:{shorthand}"
    else:
        semitones = CHORD_DEGREES[quality][bass_degree]
        bass = (root + semitones) % 12
        harte = f"{root_name}:{shorthand}/{HARTE_INTERVAL_NAMES[semitones]}"

    return root, quality, bass, harte


def build_label_table():
    """Return one row per clip for every WAV in both family folders."""
    rows = []
    for family in FAMILIES:
        folder = PROJECT_DIR / family
        lab_table = read_lab(folder / f"{family}_annotation.lab")

        for wav_path in sorted(folder.glob("*.wav")):
            instrument = wav_path.stem  # e.g. 'ableton_live_guitar_Campfire'
            for index, lab_row in lab_table.iterrows():
                root, quality, bass, harte = parse_label(lab_row["label"])
                rows.append({
                    "clip_id": f"{instrument}_{index:03d}",
                    "source_file": f"{family}/{wav_path.name}",
                    "instrument": instrument,
                    "family": family,
                    "start_s": lab_row["start_s"],
                    "end_s": lab_row["end_s"],
                    "root": root,
                    "quality": quality,
                    "bass": bass,
                    "harte": harte,
                    "original_label": lab_row["label"],
                })
    return pd.DataFrame(rows)


def main():
    labels = build_label_table()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")

    # Checks for Step 1
    print(f"\nTotal rows: {len(labels)}")
    print("\nClips per root:")
    print(labels["root"].map(lambda r: NOTE_NAMES[r]).value_counts().reindex(NOTE_NAMES).to_string())
    print("\nClips per quality:")
    print(labels["quality"].value_counts().to_string())
    print("\nRoot position vs inversion, by family:")
    print(pd.crosstab(labels["family"], labels["bass"] == labels["root"]).rename(
        columns={True: "root position", False: "inversion"}).to_string())
    print("\n5 random rows (to listen to):")
    pd.set_option("display.width", 200)
    print(labels.sample(5, random_state=SEED).to_string(index=False))


if __name__ == "__main__":
    main()
