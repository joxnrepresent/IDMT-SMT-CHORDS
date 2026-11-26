# src/check_power.py : which power chord does each guitar "power" clip look most like?
import json
from pathlib import Path
import numpy as np, pandas as pd
from features import PC

d = pd.read_csv("src/data/processed/labels.csv")
X = np.load("data/features.npy")[:, :12]
test = json.loads(next(Path(".").rglob("splits.json")).read_text(encoding="utf-8-sig"))["test"]
keep = ((d.quality == "power") & (d.family == "guitar") & ~d.instrument.isin(test)).to_numpy()
p, X = d[keep].copy(), X[keep]

# energy on (root, fifth) for all 12 possible roots; pick the strongest
looks = np.argmax([X[:, r] + X[:, (r + 7) % 12] for r in range(12)], axis=0)
p["semitones_above_label"] = (looks - p.root.to_numpy()) % 12
p["looks_like"] = [PC[r] for r in looks]

print("Number of clips at each distance from the label (0 = matches):")
print(pd.crosstab(p.instrument, p.semitones_above_label).to_string())
first = p[p.instrument == p.instrument.iloc[0]]
print("\nEvery power clip for", first.instrument.iloc[0])
print(first[["start_s", "original_label", "looks_like", "semitones_above_label"]].to_string(index=False))
