"""Plot chord-aligned chroma features for the Ableton Campfire guitar recording."""
import csv
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import Patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_DIR / "data" / "audio"
DEFAULT_ANNOTATION_PATH = AUDIO_DIR / "guitar" / "guitar_annotation.lab"
OUTPUT_DIR = PROJECT_DIR / "plots" / "Chroma features"
FAMILIES = ("guitar", "non_guitar")

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITY_NAMES = {
    "maj": "Major",
    "min": "Minor",
    "maj7": "Major 7",
    "min7": "Minor 7",
    "5": "Power",
    "7": "Dominant 7",
    "hdim7": "Half-diminished 7",
}


def read_annotations(path=DEFAULT_ANNOTATION_PATH):
    with path.open(newline="") as annotation_file:
        return [
            (float(start), float(end), label.strip())
            for start, end, label in csv.reader(annotation_file, delimiter="\t")
        ]


def chroma_frames_for_interval(audio, sample_rate, start, end):
    segment = audio[round(start * sample_rate):round(end * sample_rate)]
    chroma = librosa.feature.chroma_stft(
        y=segment,
        sr=sample_rate,
        n_fft=8192,
        hop_length=2048,
    )
    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]),
        sr=sample_rate,
        hop_length=2048,
    )
    stable_frames = (frame_times >= 0.15) & (frame_times < end - start - 0.15)
    if not stable_frames.any():
        raise ValueError(f"No stable chroma frames found for chord at {start} s")
    return chroma[:, stable_frames]


def compute_chroma_for_interval(audio, sample_rate, start, end):
    return chroma_frames_for_interval(audio, sample_rate, start, end).mean(axis=1)


def compute_chroma_by_chord(audio, sample_rate, annotations):
    features = []
    chords_per_batch = 15
    for batch_start in range(0, len(annotations), chords_per_batch):
        batch = annotations[batch_start:batch_start + chords_per_batch]
        start, end = batch[0][0], batch[-1][1]
        segment = audio[round(start * sample_rate):round(end * sample_rate)]
        chroma = librosa.feature.chroma_stft(
            y=segment,
            sr=sample_rate,
            n_fft=8192,
            hop_length=2048,
        )
        frame_times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sample_rate,
            hop_length=2048,
        )
        for chord_start, chord_end, _ in batch:
            stable_frames = (
                (frame_times >= chord_start - start + 0.15)
                & (frame_times < chord_end - start - 0.15)
            )
            if not stable_frames.any():
                raise ValueError(f"No stable chroma frames found for chord at {chord_start} s")
            features.append(chroma[:, stable_frames].mean(axis=1))
    return np.asarray(features).T


def plot_recording(audio_path, annotation_path):
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    annotations = read_annotations(annotation_path)
    duration = len(audio) / sample_rate
    if not np.isclose(annotations[-1][1], duration):
        raise ValueError(
            f"{audio_path.name}: annotations end at {annotations[-1][1]} s "
            f"but audio is {duration} s"
        )

    chroma_by_chord = compute_chroma_by_chord(audio, sample_rate, annotations)
    chord_starts = np.asarray([row[0] for row in annotations])
    chord_ends = np.asarray([row[1] for row in annotations])
    chord_labels = [row[2] for row in annotations]
    qualities = [label.split(":", 1)[1].split("/", 1)[0] for label in chord_labels]

    fig, ax = plt.subplots(figsize=(25, 8))
    image = ax.imshow(
        chroma_by_chord,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0, duration, -0.5, len(PITCH_CLASSES) - 0.5),
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(range(len(PITCH_CLASSES)), PITCH_CLASSES)
    ax.set_ylabel("Pitch class")
    ax.set_xlabel("Time (seconds)")
    ax.set_title(
        f"Chord-aligned chroma features — {audio_path.stem.replace('_', ' ')}",
        pad=42,
        fontsize=15,
    )

    for start in chord_starts[1:]:
        ax.axvline(start, color="white", alpha=0.12, linewidth=0.45)

    quality_colors = plt.get_cmap("tab10").colors
    quality_color = {
        quality: quality_colors[index % len(quality_colors)]
        for index, quality in enumerate(QUALITY_NAMES)
    }
    group_start = 0
    for index in range(1, len(qualities) + 1):
        if index == len(qualities) or qualities[index] != qualities[group_start]:
            quality = qualities[group_start]
            start, end = chord_starts[group_start], chord_ends[index - 1]
            ax.axvspan(start, end, color=quality_color[quality], alpha=0.08, zorder=2)
            ax.axvline(start, color="white", alpha=0.75, linewidth=1.0, zorder=3)
            ax.text(
                (start + end) / 2,
                1.015,
                QUALITY_NAMES.get(quality, quality),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color=quality_color[quality],
            )
            group_start = index

    tick_indices = np.arange(0, len(annotations), 6)
    ax.set_xticks(
        chord_starts[tick_indices],
        [chord_labels[index] for index in tick_indices],
        rotation=90,
        fontsize=7,
    )
    ax.set_xlim(0, duration)
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Mean normalized chroma")
    fig.text(
        0.5,
        0.015,
        "Each column averages chroma over the stable middle of one annotated 2-second chord.",
        ha="center",
        fontsize=9,
    )
    legend = [
        Patch(facecolor=quality_color[quality], label=name)
        for quality, name in QUALITY_NAMES.items()
    ]
    ax.legend(
        handles=legend,
        title="Chord quality",
        loc="upper right",
        ncol=2,
        fontsize=7,
        title_fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.95))
    output_path = OUTPUT_DIR / f"chroma_full_{audio_path.stem}.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path} ({len(annotations)} chords, {duration:.1f} s)")


def main():
    n_plots = 0
    for family in FAMILIES:
        family_dir = AUDIO_DIR / family
        annotation_path = family_dir / f"{family}_annotation.lab"
        for audio_path in sorted(family_dir.glob("*.wav")):
            plot_recording(audio_path, annotation_path)
            n_plots += 1
    print(f"Saved full-sequence chroma plots for {n_plots} instruments to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
