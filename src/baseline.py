# src/baseline.py
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from features import QUALITIES, PC

df = pd.read_csv("src/data/processed/labels.csv")
C = np.load("data/features.npy")[:, :12]
assert len(df) == len(C), "labels.csv and features.npy have different numbers of rows"

# leave out the 3 test instruments until Stage 5
found = list(Path(".").rglob("splits.json"))
if len(found) != 1:
    raise SystemExit(f"Expected one splits.json, found: {found}")
test = json.loads(found[0].read_text(encoding="utf-8-sig"))["test"]
unknown = set(test) - set(df.instrument)
if unknown:
    raise SystemExit(f"Not in labels.csv: {unknown}\nlabels.csv has: {sorted(df.instrument.unique())}")
dev = ~df.instrument.isin(test).to_numpy()
df, C = df[dev].reset_index(drop=True), C[dev]
print(f"Using {len(df)} clips from {df.instrument.nunique()} instruments")
Path("results").mkdir(exist_ok=True)

# Fig 3: average chroma of each type, with every clip moved to C
root_idx = df.root.to_numpy(dtype=int)                  # labels.csv stores roots as 0-11, C = 0
assert set(df.quality) == set(QUALITIES), f"type names differ: {sorted(set(df.quality))}"
rot = np.array([np.roll(c, -r) for c, r in zip(C, root_idx)])
fig, axes = plt.subplots(1, 7, figsize=(21, 3), sharey=True)
for ax, q in zip(axes, QUALITIES):
    ax.bar(PC, rot[(df.quality == q).to_numpy()].mean(axis=0)); ax.set_title(f"C:{q}")
plt.tight_layout(); plt.savefig("results/fig3_avg_chroma.png", dpi=150); plt.close()

# template baseline
names, T = [], []
for r in range(12):
    for q, ivs in QUALITIES.items():
        t = np.zeros(12); t[[(r + i) % 12 for i in ivs]] = 1
        T.append(t / np.linalg.norm(t)); names.append((r, q))
T = np.array(T)                                                   # 84 x 12
Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
best = np.argmax(Cn @ T.T, axis=1)                                # cosine similarity
pred_root = np.array([names[b][0] for b in best])
pred_q = np.array([names[b][1] for b in best])

ok_root, ok_q = pred_root == root_idx, pred_q == df.quality.to_numpy()
scores = pd.DataFrame([dict(root=ok_root.mean(), type=ok_q.mean(), both=(ok_root & ok_q).mean())])
scores.to_csv("results/baseline_scores.csv", index=False)
print(scores.round(3).to_string(index=False))

# Fig 4: true type against guessed type
ConfusionMatrixDisplay.from_predictions(df.quality, pred_q, normalize="true", values_format=".2f")
plt.tight_layout(); plt.savefig("results/fig4_baseline_confusion.png", dpi=150); plt.close()

# top 5 mistakes
true_lab = [f"{PC[r]}:{q}" for r, q in zip(root_idx, df.quality)]
pred_lab = [f"{PC[r]}:{q}" for r, q in zip(pred_root, pred_q)]
top = pd.DataFrame({"true": true_lab, "pred": pred_lab}).query("true != pred").value_counts().head(5)
top.to_csv("results/baseline_top_mistakes.csv")
print("\nTop 5 mistakes:")
print(top.to_string())
