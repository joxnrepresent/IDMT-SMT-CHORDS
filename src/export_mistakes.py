# src/export_mistakes.py : save 2 example clips for each of the top 5 baseline mistakes
import json
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
from features import QUALITIES, PC, SR

df = pd.read_csv("src/data/processed/labels.csv")
C = np.load("data/features.npy")[:, :12]
clips = np.load("data/clips_16k.npy", mmap_mode="r")
test = json.loads(next(Path(".").rglob("splits.json")).read_text(encoding="utf-8-sig"))["test"]
dev = ~df.instrument.isin(test).to_numpy()

# same template guesses as baseline.py
names, T = [], []
for r in range(12):
    for q, ivs in QUALITIES.items():
        t = np.zeros(12); t[[(r + i) % 12 for i in ivs]] = 1
        T.append(t / np.linalg.norm(t)); names.append(f"{PC[r]}:{q}")
Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
df["pred"] = np.array(names)[np.argmax(Cn @ np.array(T).T, axis=1)]
df["true"] = [f"{PC[int(r)]}:{q}" for r, q in zip(df.root, df.quality)]

top = df[dev & (df.true != df.pred).to_numpy()][["true", "pred"]].value_counts().head(5)
out = Path("results/listen"); out.mkdir(parents=True, exist_ok=True)
rng, rows = np.random.default_rng(0), []

for n, (true, pred) in enumerate(top.index, 1):
    wrong = df[dev & ((df.true == true) & (df.pred == pred)).to_numpy()]
    picks = wrong.iloc[rng.permutation(len(wrong))].drop_duplicates("instrument").head(2)
    for k, (i, row) in enumerate(picks.iterrows(), 1):
        name = f"mistake{n}_clip{k}.wav"
        sf.write(out / name, clips[i].astype(np.float32), SR)
        r = int(row.root)
        rows.append(dict(file=name, true=true, guessed=pred, instrument=row.instrument,
                         root=round(float(C[i, r]), 2), third=round(float(C[i, (r + 4) % 12]), 2),
                         fifth=round(float(C[i, (r + 7) % 12]), 2), what_i_heard=""))

table = pd.DataFrame(rows)
table.to_csv(out / "notes.csv", index=False)
print(table.drop(columns="what_i_heard").to_string(index=False))
