# src/make_clips.py
from pathlib import Path
import numpy as np, pandas as pd, librosa

SR = 16000
AUDIO = Path.home() / "Downloads" / "IDMT-SMT-CHORDS" / "IDMT-SMT-CHORDS"

labels = pd.read_csv("src/data/processed/labels.csv").reset_index(drop=True)
wavs = {p.name: p for p in AUDIO.rglob("*.wav")}

# stop early if any file named in labels.csv cannot be found
missing = [s for s in labels.source_file.unique() if Path(s).name not in wavs]
if missing:
    raise SystemExit(f"Could not find these audio files: {missing}")

clips = np.zeros((len(labels), 2 * SR), dtype=np.float16)

for src, rows in labels.groupby("source_file"):
    print("Loading", src, flush=True)
    y, _ = librosa.load(wavs[Path(src).name], sr=SR, mono=True)   # resample once per file
    for i, r in rows.iterrows():
        a = int(round(r.start_s * SR))
        seg = y[a : a + 2 * SR]
        clips[i, : len(seg)] = seg

Path("data").mkdir(exist_ok=True)
np.save("data/clips_16k.npy", clips)                # row i = labels.csv row i

silent = int((np.abs(clips).max(axis=1) == 0).sum())
print("Done. Shape:", clips.shape, "| completely silent clips:", silent)
