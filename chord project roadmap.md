# Chord Recognition Project: Roadmap

## What we are building

A system that hears 2 seconds of audio and names the chord: its root (C to B) and its type (7 types), so 84 chords in total. The brief's pipeline: load audio, extract features, train, evaluate.

On top of that we answer one research question:

> **Can a 12-number "overtone fingerprint" of an instrument predict a chord classifier's mistakes on that instrument, and can a reinforcement-learning agent use the same fingerprint to choose training data that prevents them?**

Constraints from the brief: clips under 5 s, audio at 16 kHz, dataset under 2 GB, scikit-learn models plus one small PyTorch CNN.

## Timeline

| Stage | Due |
|---|---|
| 1. Data | Sun 15 Nov |
| 2. Dataset analysis | Tue 17 Nov |
| 3. Features and baseline | Thu 19 Nov |
| 4. Models | Sun 22 Nov |
| 5. Evaluation | Tue 24 Nov |
| 6. Extension | E1 Fri 20 Nov · E2 Mon 23 Nov · E3 Fri 27 Nov |
| 7. Presentation | slides Sun 29 Nov · rehearsal Mon 30 Nov · **talk Tue 1 Dec, 09:00** |
| 8. Individual reports | outline Thu 3 Dec · draft Mon 7 Dec · submit Wed 9 Dec (**deadline Thu 10 Dec, 12:00**) |

Stage 6 needs only the chroma from Stage 3, so it runs alongside Stages 4 and 5.

## Before starting

```
data/      raw/ (the 16 WAVs), labels.csv, splits.json, clips_16k.npy, features.npy
src/       features.py, fingerprint.py, bandit.py, one script per step
results/   every figure (PNG) and table (CSV), each made by a script
models/    final.joblib (plus cnn.pt if the CNN is kept)
docs/      log.md (dated: what we tried, the result, the decision)
```

Git-ignore `data/` except `labels.csv` and `splits.json`: the rest is large and can be rebuilt.

---

## Stage 1: Data (due Sun 15 Nov)

`labels.csv` is done and checked: 7,398 rows, one per 2-second chord. This stage turns the 16 WAVs into one array of clips whose rows line up with `labels.csv`.

**1. Slice the files.** Load each WAV once at 16 kHz mono and take 2 s from each label's start time. Stored as float16 this is about 470 MB.

```python
# src/make_clips.py
import numpy as np, pandas as pd, librosa

SR = 16000
labels = pd.read_csv("data/processed/labels.csv").reset_index(drop=True)
clips = np.zeros((len(labels), 2 * SR), dtype=np.float16)

for src, rows in labels.groupby("source_file"):
    y, _ = librosa.load(f"data/raw/{src}", sr=SR, mono=True)   # resample once per file
    for i, r in rows.iterrows():
        a = int(round(r.start_s * SR))
        seg = y[a : a + 2 * SR]
        clips[i, : len(seg)] = seg

np.save("data/clips_16k.npy", clips)                             # row i = labels.csv row i
```

**2. Check the counts.** You should get 7,398 clips: 576 in each of the 10 non-guitar files (called "keyboard" below) and 273 in each of the 6 guitar files. By type: maj7, min7, dom7 and m7♭5 have 1,194 each, maj and min 954 each, and power 714. Any other number means a slicing bug.

---

## Stage 2: Dataset analysis (due Tue 17 Nov)

**1. Counts → Fig 1.** Bar charts of clips per type, root, family and inversion.

```python
import numpy as np, pandas as pd, librosa, librosa.display, matplotlib.pyplot as plt
SR = 16000
df = pd.read_csv("data/processed/labels.csv")
clips = np.load("data/processed/clips_16k.npy").astype(np.float32)

df["inverted"] = df.harte.str.contains("/")        # Harte labels mark inversions with "/"
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, col in zip(axes, ["quality", "root", "family", "inverted"]):
    df[col].value_counts().sort_index().plot.bar(ax=ax, title=col)
plt.tight_layout(); plt.savefig("results/fig1_counts.png", dpi=150)
pd.crosstab(df.instrument, df.quality).to_csv("results/table_counts.csv")
```

What to note: power chords are the smallest class (714 against 1,194), so we report macro-F1 as well as accuracy. Only the keyboard files contain inversions.

**2. One chord on three instruments → Fig 2.** A constant-Q spectrogram has one row per note, so overtones show up as extra notes.

```python
chord = "C:maj"
insts = df.instrument.unique()[[0, 5, 12]]        # pick three that sound different
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, inst in zip(axes, insts):
    i = df[(df.harte == chord) & (df.instrument == inst)].index[0]
    C = np.abs(librosa.cqt(clips[i], sr=SR, hop_length=256,
                           fmin=librosa.note_to_hz("C1"), n_bins=84))
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=SR, hop_length=256,
                             fmin=librosa.note_to_hz("C1"), x_axis="time", y_axis="cqt_note", ax=ax)
    ax.set_title(f"{chord} on {inst}")
plt.tight_layout(); plt.savefig("results/fig2_cqt.png", dpi=150)
```

**3. Brightness per instrument → Table 1.** The spectral centroid is the "centre of mass" of the spectrum: higher means brighter, which means stronger overtones.

```python
df["centroid_hz"] = [librosa.feature.spectral_centroid(y=x, sr=SR)[0].mean() for x in clips]
t1 = (df.groupby(["instrument", "family"]).centroid_hz
        .agg(median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
        .sort_values("median"))
t1.round(0).to_csv("results/table1_brightness.csv")
```

**4. Choose 3 test instruments → `data/splits.json`.** Pick 2 keyboard-type and 1 guitar, spread across Table 1 (one bright, one dark, one in the middle; avoid the single most extreme). For example: `{"test": ["...", "...", "..."]}`. From now on nobody trains on them, tunes on them or looks at results for them until Stage 5.

**5. Check the repeats.** Each keyboard file plays every chord twice under the same label. Listen to 3 pairs and confirm what changes (probably the octave). Either way, both plays of a chord stay on the same side of every split (Stage 4).

**Output:** Fig 1, Fig 2, Table 1, `splits.json`.
**For the report (Datasets):** class balance and the macro-F1 choice; how the instruments differ (Fig 2, Table 1); how the test instruments were chosen.

---

## Stage 3: Features and baseline (due Thu 19 Nov)

**Chroma** is how a clip's energy is spread over the 12 pitch classes (C, C#, …, B), with all octaves folded together. A chord is a set of pitch classes, so this is the natural feature. **Bass chroma** is the same thing using only 40–250 Hz, which helps find the lowest note.

**1. Write the features with our own FFT.**

```python
# src/features.py
import numpy as np

SR, ATTACK_S = 16000, 0.10           # set ATTACK_S from Stage 1
N_FFT, HOP = 8192, 2048              # 8,192 points at 16 kHz = 1.95 Hz per bin
f = np.fft.rfftfreq(N_FFT, 1 / SR)
midi = 69 + 12 * np.log2(np.maximum(f, 1e-6) / 440.0)
pc = np.round(midi).astype(int) % 12              # pitch class of every FFT bin, C = 0
MAIN = (f >= 55) & (f <= 5000)                    # A1 to about D#8
BASS = (f >= 40) & (f <= 250)                     # about E1 to B3
WIN = np.hanning(N_FFT)

def power_spectrum(x):
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * WIN
    return (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)   # average over frames

def features(clip):
    """2 s of audio at 16 kHz -> 24 numbers: chroma (12), then bass chroma (12)."""
    x = clip[int(ATTACK_S * SR):].astype(np.float64)
    P = power_spectrum(x)
    chroma = np.bincount(pc[MAIN], weights=P[MAIN], minlength=12)
    bass = np.bincount(pc[BASS], weights=P[BASS], minlength=12)
    total = chroma.sum() + 1e-12
    return np.concatenate([chroma / total, bass / total])   # same divisor keeps "how much bass"
```

How it works: Hann-windowed frames of 8,192 samples (0.51 s, fine enough to separate semitones down to about A1), a power spectrum averaged over the frames, then each FFT bin is added to its nearest pitch class. Both halves are divided by the same total, so the bass half still says how much bass there is. Run it on every clip and save the result (7,398 × 24):

```python
# src/make_features.py
import numpy as np
from features import features
clips = np.load("data/clips_16k.npy").astype(np.float32)
np.save("data/features.npy", np.array([features(c) for c in clips]))
```

**2. Check against librosa.** On 5 random clips, our chroma should correlate highly with librosa's (above about 0.9). It won't match exactly, because librosa weights the octaves differently.

```python
ref = librosa.feature.chroma_stft(y=clip[int(ATTACK_S*SR):], sr=SR, n_fft=8192,
                                  hop_length=2048, tuning=0).mean(axis=1)
print(np.corrcoef(features(clip)[:12], ref)[0, 1])
```

**3. Average chroma of each type → Fig 3.** Transpose every clip to C, then average by type. This is the key picture of the project: energy on notes outside the chord (for example B in C major) is overtone leakage, and Stage 6 starts from it.

Add these three constants to the bottom of `src/features.py`; later stages import them.

```python
ROOTS = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
         "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
QUALITIES = {"maj":[0,4,7], "min":[0,3,7], "maj7":[0,4,7,11], "min7":[0,3,7,10],
             "dom7":[0,4,7,10], "m7b5":[0,3,6,10], "power":[0,7]}      # keys must match labels.csv
PC = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
```

```python
# src/baseline.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from features import ROOTS, QUALITIES, PC

df = pd.read_csv("data/labels.csv")
C = np.load("data/features.npy")[:, :12]
root_idx = df.root.map(ROOTS).to_numpy()
rot = np.array([np.roll(c, -r) for c, r in zip(C, root_idx)])     # every clip moved to C
fig, axes = plt.subplots(1, 7, figsize=(21, 3), sharey=True)
for ax, q in zip(axes, QUALITIES):
    ax.bar(PC, rot[df.quality == q].mean(axis=0)); ax.set_title(f"C:{q}")
plt.tight_layout(); plt.savefig("results/fig3_avg_chroma.png", dpi=150)
```

**4. Template baseline → Fig 4.** Build a template for each of the 84 chords (1 on its notes, 0 elsewhere) and pick the template most similar to the clip's chroma. There is no learning in it, so it is the score that learning has to beat.

```python
names, T = [], []
for r in range(12):
    for q, ivs in QUALITIES.items():
        t = np.zeros(12); t[[(r + i) % 12 for i in ivs]] = 1
        T.append(t / np.linalg.norm(t)); names.append((r, q))
T = np.array(T)                                                   # 84 x 12
Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
best = np.argmax(Cn @ T.T, axis=1)                                # cosine similarity
pred_root = np.array([names[b][0] for b in best])
pred_q = np.array([names[b][1] for b in best])

ok_root, ok_q = pred_root == root_idx, pred_q == df.quality.to_numpy()
print(ok_root.mean(), ok_q.mean(), (ok_root & ok_q).mean())       # root, type, both

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(df.quality, pred_q, normalize="true", values_format=".2f")
plt.savefig("results/fig4_baseline_confusion.png", dpi=150)
```

**5. Explain the top 5 mistakes with music theory.**

```python
true_lab = [f"{PC[r]}:{q}" for r, q in zip(root_idx, df.quality)]
pred_lab = [f"{PC[r]}:{q}" for r, q in zip(pred_root, pred_q)]
print(pd.DataFrame({"true": true_lab, "pred": pred_lab}).query("true != pred").value_counts().head(5))
```

Most mistakes will be subset relations. Every 7th chord contains a triad: Cmaj7 contains C and Em, C7 contains C, Cm7 contains Cm and E♭, Cm7♭5 contains E♭m. Also, C is contained in Am7. For each mistake, listen to two clips and name the note that is too loud or too quiet.

**Output:** `features.npy`, Figs 3 and 4, baseline accuracy for root, type and both.
**For the report:** Methods (chroma, the baseline); Introduction (why learning is needed: the baseline's mistakes).

---

## Stage 4: Models (due Sun 22 Nov)

**Predict the 84 chords jointly, not root and type separately.** A linear model struggles to find "major" in raw chroma, because a major chord is a different pattern in each of the 12 keys. In our code check on synthetic chords, a type-only logistic regression scored near chance while the joint 84-class one was fine. Read root and type accuracy off the predicted chord. Try the separate version once on the real data: it is a good example of linear separability for the report.

**Train with seen and unseen instruments: yes, do both.** Use the same models and the same settings, and split the 13 development instruments in two ways:

| Condition | Splitter | Question it answers |
|---|---|---|
| Seen instruments | `StratifiedGroupKFold(5)`, groups = instrument + chord, so both plays of a chord stay together | How good is it on instruments it has heard? |
| Unseen instruments | `GroupKFold(5)`, groups = instrument | How good is it on an instrument it has never heard? |
| Held-out test (Stage 5, once) | Train on all 13, test on the 3 in `splits.json` | The final, untouched check |

The gap between the first two rows is a headline result: a random split flatters the model. After Stage 5, train the final model on all 16 instruments for `infer.py` and the keyboard test.

```python
# src/evaluate_cv.py
import numpy as np, pandas as pd, json
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import f1_score

df = pd.read_csv("data/labels.csv"); X = np.load("data/features.npy")
test_instruments = json.load(open("data/splits.json"))["test"]
dev = ~df.instrument.isin(test_instruments).to_numpy()
Xd, d = X[dev], df[dev].reset_index(drop=True)
chord = (d.root + ":" + d.quality).to_numpy()

CONDITIONS = {
    "seen":   (StratifiedGroupKFold(5, shuffle=True, random_state=0), d.instrument + "|" + d.harte),
    "unseen": (GroupKFold(5), d.instrument),
}

def evaluate(make_model):
    out = {}
    for name, (cv, groups) in CONDITIONS.items():
        folds = []
        for tr, te in cv.split(Xd, chord, groups):
            pred = make_model().fit(Xd[tr], chord[tr]).predict(Xd[te])
            pr, pq = np.array([p.split(":") for p in pred]).T
            ok_r, ok_q = pr == d.root[te].to_numpy(), pq == d.quality[te].to_numpy()
            folds.append(dict(root=ok_r.mean(), type=ok_q.mean(), both=(ok_r & ok_q).mean(),
                              type_f1=f1_score(d.quality[te], pq, average="macro")))
        f = pd.DataFrame(folds)
        out[name] = {k: f"{f[k].mean():.3f} ± {f[k].std():.3f}" for k in f}
    return out
```

**Models and tuning.** Tune each model with `GridSearchCV` using unseen-instrument folds on the 13 development instruments (that is the real use case), then run `evaluate` with the chosen settings.

| Model | Grid | Why it is here |
|---|---|---|
| Logistic regression | C ∈ {0.1, 1, 10, 100} | Linear; gives probabilities, which Stage 6 uses |
| k-NN | k ∈ {1, 3, 5, 9, 15}; weights ∈ {uniform, distance} | A learned version of template matching |
| SVM (RBF kernel) | C ∈ {1, 10, 100}; gamma ∈ {scale, 0.01, 0.1} | Non-linear, margin-based |
| Random forest | 300 trees; max_depth ∈ {None, 20}; max_features ∈ {sqrt, 0.5} | Non-linear, needs no scaling, shows feature importance |

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone

search = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                      {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]},
                      cv=GroupKFold(5), scoring="accuracy")
search.fit(Xd, chord, groups=d.instrument)
print(search.best_params_)
print(evaluate(lambda: clone(search.best_estimator_)))
```

Put every model except the random forest behind `StandardScaler`.

**Small CNN on a constant-Q spectrogram.** Input: log CQT, 84 note rows (C1 to B7) × about 119 frames. Pool over time only, so the note position (which carries the root) survives.

```python
import numpy as np, librosa, torch, torch.nn as nn
from features import SR, ATTACK_S

def cqt_input(clip):
    C = np.abs(librosa.cqt(clip[int(ATTACK_S * SR):], sr=SR, hop_length=256,
                           fmin=librosa.note_to_hz("C1"), n_bins=84))
    return np.log1p(100 * C).astype(np.float32)                # 84 x ~119

class ChordCNN(nn.Module):
    def __init__(self, n_classes=84):
        super().__init__()
        block = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.net = nn.Sequential(*block(1, 16), nn.MaxPool2d((1, 2)),     # pool time only
                                 *block(16, 32), nn.MaxPool2d((1, 2)),
                                 *block(32, 32), nn.AdaptiveAvgPool2d((84, 1)))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(32 * 84, n_classes))
    def forward(self, x):                                      # x: (batch, 1, 84, frames)
        return self.head(self.net(x))                          # about 240k parameters
```

Train with Adam (learning rate 1e-3), batch size 64 and cross-entropy for up to 30 epochs. Keep the epoch with the best validation accuracy, using one training instrument as validation (unseen condition) or 10% of the training clips (seen condition). Use the same two conditions and folds as the other models. **Fig 5:** training and validation curves for one fold; say whether it overfits.

**Log every failed attempt** in `docs/log.md`: the date, what was tried, the result and the decision.

**Output:** Table 2 (baseline and all 5 models; seen and unseen; root, type and both accuracy plus type macro-F1, each as mean ± std over folds); Fig 5; the chosen settings.
**For the report:** Methods (models, tuning, the split); Evaluation (Table 2).

---

## Stage 5: Evaluation (due Tue 24 Nov)

**1. Held-out test, once.** Train each model on all 13 development instruments with its tuned settings and score it on the 3 test instruments. **Table 3:** held-out accuracy per model. **Fig 6:** for each model, three bars: seen (CV), unseen (CV) and held-out.

**2. Look closely at the best model.** **Fig 7a:** its confusion matrix over the 7 types. **Fig 7b:** accuracy per instrument against brightness (Table 1). If bright instruments do worse, that is the motivation for Stage 6 in one picture.

**3. Listen to 10 wrong clips** and note what they share: instrument, type pair, register? This is the subjective listening part of the evaluation the brief asks for.

**4. Record a keyboard test set.** In the DAW, record audio and MIDI together: all 84 chords in root position, 2 s each with a gap, metronome at 60 bpm. Use a sound that is not in the dataset; a microphone in a room is a harder and more realistic test if you can manage it. Export audio and MIDI from the same start point so the times line up. Label from the MIDI:

```python
import pretty_midi, pandas as pd
from features import QUALITIES
SETS = {frozenset((r + i) % 12 for i in ivs): (r, q) for r in range(12) for q, ivs in QUALITIES.items()}
NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def label_take(midi_path, gap=0.05):
    notes = sorted(pretty_midi.PrettyMIDI(midi_path).instruments[0].notes, key=lambda n: n.start)
    chords, cur = [], [notes[0]]
    for n in notes[1:]:                        # notes starting within 50 ms form one chord
        if n.start - cur[0].start < gap: cur.append(n)
        else: chords.append(cur); cur = [n]
    chords.append(cur)
    rows = []
    for c in chords:
        root, q = SETS.get(frozenset(n.pitch % 12 for n in c), (None, None))  # None: not one of our 84
        rows.append(dict(start_s=min(n.start for n in c), end_s=max(n.end for n in c),
                         root=NAMES[root] if root is not None else None, quality=q))
    return pd.DataFrame(rows)
```

Slice the audio at those start times, compute the same features, and score the final model. **Table 4:** accuracy on our recordings next to the dataset scores.

**5. Final model and `infer.py`.** Train the best model on all 16 instruments and save it with `joblib.dump(model, "models/final.joblib")`. The brief requires a script that loads the model and names the chord in a .wav file:

```python
# infer.py   usage: python infer.py chord.wav   ->   prints e.g. C:maj7
import sys, joblib, librosa, numpy as np
sys.path.insert(0, "src")
from features import features, SR

y, _ = librosa.load(sys.argv[1], sr=SR, mono=True)
clip = np.pad(y[: 2 * SR], (0, max(0, 2 * SR - len(y))))    # first 2 s, padded if shorter
model = joblib.load("models/final.joblib")
print(model.predict(features(clip)[None])[0])
```

If the CNN wins, ship it instead (`models/cnn.pt` plus the CQT path in `infer.py`).

**6. Export 6 clips for the talk:** 3 the model gets right and 3 it gets wrong, each with one line on what it said and the likely reason.

**7. Write the README:** how to install, download the dataset from Zenodo, rebuild the data, run each step and run `infer.py`.

**Output:** Figs 6–7, Tables 3–4, `models/final.joblib`, `infer.py`, README, 6 audio clips.
**For the report:** Evaluation.

---

## Stage 6: Extension (E1 Fri 20 Nov · E2 Mon 23 Nov · E3 Fri 27 Nov)

### The idea

- **Leakage.** Every note also puts energy on other pitch classes through its overtones. Harmonics 3 and 6 land on the note's fifth, 5 and 10 on its major third, and 7 near its minor seventh (all an octave or more up). In a bright C major (C E G), E's fifth and G's major third both land on B, so the chord starts to look like Cmaj7.
- **Fingerprint.** We describe an instrument by 12 numbers *h*: the share of one note's energy that lands on each pitch class, measured from that note. So *h*[0] is the note and its octaves, *h*[7] a fifth up, *h*[4] a major third up, and so on. A chord's chroma is then the average of *h* shifted to each of its notes. That is linear in *h*, so *h* can be fitted from labelled chords with non-negative least squares (NNLS).
- **Predict.** Fit *h* for a new instrument from 3 labelled chords, generate the chroma that all 84 chords should have on it, and run the trained model on those. This predicts the model's accuracy and its confusions without classifying any real clip of the new instrument beyond those 3.
- **Train with RL.** The same formula with a made-up *h* is a one-line instrument simulator. A bandit agent chooses which made-up instruments to train on, and its reward is how much the classifier learns from them.

### Hypotheses

Commit these to the repo, dated, before running E2 and E3:

- **H1.** One fingerprint per instrument explains most of its chroma (high R²), and brighter instruments have larger *h*[7], *h*[4] and *h*[10].
- **H2.** Fingerprint predictions of accuracy on a new instrument are closer to the truth than ATC, the standard confidence-based method, and than testing on the 3 calibration chords.
- **H3.** Adding simulated instruments, chosen uniformly at random, raises accuracy on unseen real instruments over training on the home instrument alone.
- **H4.** The bandit reaches uniform sampling's final accuracy with fewer simulated chords, because it spends its budget on bright or blurred instruments early.

Each can fail. A failure is still a result if we can say why.

### Shared setup

- **Features:** the 12 chroma numbers only. The fingerprint says nothing about bass chroma.
- **Classifier:** `make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=3000))` over the 84 chords. Scale first: chroma values are small, and without scaling a regularised logistic regression is very under-confident, which breaks the ATC baseline.
- **Protocol:** train on one instrument (the "home") and test on another (the "target").
- **Data hygiene:** develop and debug on the 13 development instruments only. Once code and settings are frozen, run the final E2 and E3 on all 16.
- **Repeats:** 5 seeds or draws; report mean ± std.

### E1: Fit the fingerprints (half a day, due Fri 20 Nov)

```python
# src/fingerprint.py
import numpy as np
from scipy.optimize import nnls
from features import QUALITIES

CHORDS = [(r, q) for r in range(12) for q in QUALITIES]       # the 84 classes

def notes_vector(root, quality):
    x = np.zeros(12); x[[(root + i) % 12 for i in QUALITIES[quality]]] = 1
    return x

def mix_matrix(x):
    """M with M @ h = average over the chord's notes n of h shifted up by n."""
    return np.stack([np.roll(x, d) for d in range(12)], axis=1) / x.sum()

def fit_fingerprint(chromas, xs):
    """chromas: n x 12 (each sums to 1); xs: n x 12 note vectors -> h, 12 numbers summing to 1."""
    A = np.vstack([mix_matrix(x) for x in xs])
    h, _ = nnls(A, np.concatenate(chromas))
    return h / h.sum()

def model_chroma(x, h):
    return mix_matrix(x) @ h

def fit_r2(chromas, xs, h):
    pred = np.array([model_chroma(x, h) for x in xs])
    return 1 - ((chromas - pred) ** 2).sum() / ((chromas - chromas.mean(0)) ** 2).sum()
```

Fit *h* for each of the 16 instruments from all its clips and compute R².

**Output:** **Fig 8**, the 16 fingerprints as a heatmap (instruments sorted by brightness × 12 intervals); **Table 5**, R² per instrument plus its (α, blur) position from E3's `fit_alpha_blur`.
**Check:** *h*[7] and *h*[4] should grow with brightness from Table 1. If R² is low for an instrument, listen to it: reverb, distortion or a sound that changes with register breaks the one-fingerprint assumption, and that is worth a sentence in the report.

### E2: Predict the mistakes (1.5 days, due Mon 23 Nov)

For each home *i* and each target *j* ≠ *i* (240 pairs on 16 instruments):

1. Train the classifier on home *i*. Fit the home's fingerprint and keep its **residuals** (real chroma minus fitted chroma): they are the clip-to-clip noise the simulator adds.
2. Draw 3 random labelled clips of target *j* (the calibration chords) and fit *h_j* from them.
3. **Fingerprint prediction:** simulate 5 chromas per chord from *h_j* plus home residuals; predicted accuracy is the model's accuracy on them, and predicted confusions are their confusion matrix.
4. **Baselines:** ATC (a confidence threshold set on the home instrument, applied to all unlabelled target clips) and three-chord accuracy (the model's score on the 3 calibration clips).
5. **Actual:** the model's accuracy on all target clips except the 3.
6. Repeat with 5 different calibration draws.

Add these two functions to `src/fingerprint.py`:

```python
from sklearn.model_selection import cross_val_predict, KFold

def simulate(h, residuals, rng, per_chord=5):
    """Chroma that every chord should have on an instrument with fingerprint h."""
    X, y = [], []
    for r, q in CHORDS:
        for _ in range(per_chord):
            c = model_chroma(notes_vector(r, q), h)
            c = c + np.roll(residuals[rng.integers(len(residuals))], rng.integers(12))
            c = np.clip(c, 0, None); X.append(c / c.sum()); y.append(f"{r}:{q}")
    return np.array(X), np.array(y)

def atc_threshold(model, X_home, y_home):
    """ATC (Garg et al. 2022): pick t so the share of confidences above t equals home accuracy."""
    proba = cross_val_predict(model, X_home, y_home, method="predict_proba",
                              cv=KFold(3, shuffle=True, random_state=0))
    correct = np.unique(y_home)[proba.argmax(1)] == y_home
    return np.quantile(proba.max(1), 1 - correct.mean())
```

```python
# src/e2_predict.py
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from features import ROOTS
from fingerprint import *

df = pd.read_csv("data/labels.csv")
C = np.load("data/features.npy")[:, :12]
r = df.root.map(ROOTS).to_numpy()
y = np.array([f"{a}:{q}" for a, q in zip(r, df.quality)])          # same format as simulate()
xs = np.array([notes_vector(a, q) for a, q in zip(r, df.quality)])
insts = df.instrument.unique()
rng, rows = np.random.default_rng(0), []

for home in insts:
    mh = (df.instrument == home).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=3000))
    t = atc_threshold(model, C[mh], y[mh])
    model.fit(C[mh], y[mh])
    h_home = fit_fingerprint(C[mh], xs[mh])
    resid = C[mh] - np.array([model_chroma(x, h_home) for x in xs[mh]])
    for target in insts[insts != home]:
        idx = np.flatnonzero(df.instrument == target)
        for draw in range(5):
            calib = rng.choice(idx, 3, replace=False); rest = np.setdiff1d(idx, calib)
            Xs, ys = simulate(fit_fingerprint(C[calib], xs[calib]), resid, rng)
            rows.append(dict(home=home, target=target, draw=draw,
                             actual=(model.predict(C[rest]) == y[rest]).mean(),
                             fingerprint=(model.predict(Xs) == ys).mean(),
                             atc=(model.predict_proba(C[rest]).max(1) > t).mean(),
                             three_chords=(model.predict(C[calib]) == y[calib]).mean()))
pd.DataFrame(rows).to_csv("results/e2_predictions.csv", index=False)
```

**Output:** **Fig 9**, predicted against actual accuracy (one panel per method, 240 points each); **Table 6**, mean absolute error and correlation per method (mean ± std over draws). For pairs with actual accuracy below 95%, also report how often the predicted most common type confusion (for example maj → maj7) is the real one.
**For the report:** the methods get different information (3 labelled chords against all unlabelled clips). Say so; it is part of the argument, because 3 chords are too few to test on directly but enough to fit 12 numbers.

**Go/no-go before E3.** Look at the actual accuracies from E2. If almost every pair scores above about 97%, E3 has nothing to improve. Run it only on the hardest home instruments (lowest E2 accuracy) and say why.

### E3: RL, a bandit that chooses simulated instruments (2.5 days, due Fri 27 Nov)

**The simulator.** An ideal tone whose harmonic *k* has amplitude *k*^−α (low α = bright), plus "blur" that moves some energy to the neighbouring semitones (detuning, chorus). Two numbers give a fingerprint, and the fingerprint gives the chroma of any chord.

```python
# add to src/fingerprint.py
OFFSET = {k: int(round(12 * np.log2(k))) % 12 for k in range(1, 17)}  # 3,6,12->7  5,10->4  7,14->10

def harmonic_fingerprint(alpha, blur, K=16):
    h = np.zeros(12)
    for k in range(1, K + 1):
        h[OFFSET[k]] += (k ** -alpha) ** 2                           # power, like the chroma
    h = (1 - blur) * h + blur / 2 * (np.roll(h, 1) + np.roll(h, -1))
    return h / h.sum()

def fit_alpha_blur(h_real):
    """Where a real instrument sits on the arm grid, and how far it is from it."""
    grid = [(a, b) for a in np.linspace(0.2, 4, 39) for b in np.linspace(0, 0.4, 21)]
    errs = [np.abs(harmonic_fingerprint(a, b) - h_real).sum() for a, b in grid]
    return grid[int(np.argmin(errs))], min(errs)
```

Run `fit_alpha_blur` on the 16 real fingerprints from E1. It shows where real instruments sit on the arm grid (the dots in Fig 11), and the fit error says how realistic the simulator is.

**The bandit problem**

| Part | Choice | Why |
|---|---|---|
| Arms | 21 simulated instrument types: α ∈ {0.3, 0.5, 0.75, 1, 1.5, 2, 3} × blur ∈ {0, 0.1, 0.2}, with ±10% jitter on α | Covers dull to very bright; the jitter gives variety inside an arm |
| State | None: this is a bandit, not a full MDP | The only thing that changes is the classifier, and the step size handles that |
| One round | The agent picks an arm → simulate 84 chords (one of each, plus home residual noise) → add them to the training set → refit | |
| Reward | Learning progress: log-loss on a fresh probe batch from the same arm, before minus after the refit | Rewards arms the model is learning from. Rewarding raw error makes agents fixate on arms nobody can learn (Graves et al. 2017 use signals of this kind) |
| Reward scaling | Map to [−1, 1] with the 20th and 80th percentiles of the last 100 rewards | Raw progress shrinks as training goes on |
| Agent | ε-greedy, ε = 0.1, constant step size 0.2 | The classifier changes every round, so old rewards go stale; a constant step size forgets them (Sutton & Barto, ch. 2) |
| Baselines | Uniform random arms with the same budget (standard domain randomisation); home only (no simulated data) | |
| Episode | 60 rounds = 5,040 simulated chords | About a minute with logistic regression |

```python
# src/bandit.py
import numpy as np
from fingerprint import simulate, harmonic_fingerprint
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

ARMS = [(a, b) for a in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] for b in [0.0, 0.1, 0.2]]

class FingerprintCurriculum:
    def __init__(self, X_home, y_home, resid_home, seed=0):
        self.rng = np.random.default_rng(seed)
        self.X, self.y, self.resid = X_home.copy(), y_home.copy(), resid_home
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=500))
        self.model.fit(self.X, self.y)

    def draw(self, arm):
        a, b = ARMS[arm]
        h = harmonic_fingerprint(a * self.rng.uniform(0.9, 1.1), b)
        return simulate(h, self.resid, self.rng, per_chord=1)

    def loss(self, X, y):
        return log_loss(y, self.model.predict_proba(X), labels=self.model.classes_)

    def step(self, arm):
        Xb, yb = self.draw(arm)                       # batch to train on
        Xp, yp = self.draw(arm)                       # fresh probe from the same arm
        before = self.loss(Xp, yp)
        self.X, self.y = np.vstack([self.X, Xb]), np.concatenate([self.y, yb])
        self.model.fit(self.X, self.y)
        return before - self.loss(Xp, yp)             # learning progress

class EpsGreedy:
    def __init__(self, k, eps=0.1, step=0.2, seed=0):
        self.q, self.eps, self.step = np.zeros(k), eps, step
        self.rng = np.random.default_rng(seed)
    def act(self):
        if self.rng.random() < self.eps: return self.rng.integers(len(self.q))
        return self.rng.choice(np.flatnonzero(self.q == self.q.max()))   # random tie-break
    def update(self, a, r):
        self.q[a] += self.step * (r - self.q[a])

class Uniform:
    def __init__(self, k, seed=0): self.k, self.rng = k, np.random.default_rng(seed)
    def act(self): return self.rng.integers(self.k)
    def update(self, a, r): pass

class RewardScaler:
    def __init__(self, window=100): self.hist, self.window = [], window
    def __call__(self, r):
        self.hist = (self.hist + [r])[-self.window:]
        lo, hi = np.percentile(self.hist, [20, 80])
        return 0.0 if hi == lo else float(np.clip(2 * (r - lo) / (hi - lo) - 1, -1, 1))

def run_episode(env, agent, rounds, X_targets, y_targets, eval_every=10):
    scale, choices, curve = RewardScaler(), [], []
    for t in range(rounds):
        a = agent.act()
        agent.update(a, scale(env.step(a)))
        choices.append(a)
        if (t + 1) % eval_every == 0:                 # targets are scored, never used in the reward
            curve.append(np.mean([(env.model.predict(X) == y).mean()
                                  for X, y in zip(X_targets, y_targets)]))
    return np.array(choices), np.array(curve)
```

**Protocol.** Use 4 home instruments from the development set: the dullest and the brightest keyboard and guitar in Table 1. The targets are all other real instruments, scored every 10 rounds. Run 3 conditions (home only, uniform, ε-greedy) × 4 homes × 5 seeds.

```python
# src/e3_run.py: copy the setup of e2_predict.py (imports down to "insts = ..."), then:
from bandit import *
HOMES = ["...", "...", "...", "..."]                  # from Table 1
runs = []
for home in HOMES:
    mh = (df.instrument == home).to_numpy()
    h_home = fit_fingerprint(C[mh], xs[mh])
    resid = C[mh] - np.array([model_chroma(x, h_home) for x in xs[mh]])
    Xt = [C[(df.instrument == t).to_numpy()] for t in insts if t != home]
    yt = [y[(df.instrument == t).to_numpy()] for t in insts if t != home]
    for seed in range(5):
        for name, Agent in [("uniform", Uniform), ("eps_greedy", EpsGreedy)]:
            env = FingerprintCurriculum(C[mh], y[mh], resid, seed=seed)
            home_only = np.mean([(env.model.predict(X) == Y).mean() for X, Y in zip(Xt, yt)])
            choices, curve = run_episode(env, Agent(len(ARMS), seed=seed), 60, Xt, yt)
            runs.append(dict(home=home, seed=seed, agent=name, home_only=home_only,
                             curve=curve.tolist(), choices=choices.tolist()))
pd.DataFrame(runs).to_json("results/e3_runs.json")
```

**Making it work.** Go in this order; each step is a gate for the next:

1. **Test the agent alone** on a toy bandit: 6 arms with fixed win probabilities, one clearly best, rewards of ±1. ε-greedy should pick the best arm in more than 85% of the last 500 of 2,000 rounds. If not, the bug is in the agent, not the audio.
2. **Check the simulator by eye.** For one bright real instrument, plot real against simulated chroma for the same chord, using its fitted (α, blur).
3. **Run uniform against home only first (H3).** If uniformly chosen simulated data does not help, no agent will. Stop there and report H3 as rejected, with the reason.
4. **Run a fixed-arm sweep:** train with one arm only, for each of the 21 arms (2 homes, 1 seed). If every arm ends at the same accuracy, uniform is already optimal and a bandit can only tie: report that. If some arms are clearly better, the bandit has something to find.
5. **Now run the bandit.** Log every round: arm, raw reward, scaled reward. Plot reward per arm over the rounds; it shows whether there is any signal to learn from.
6. **If it locks onto one arm early,** raise ε to 0.2 or the step size to 0.3, or change the reward to progress on a fixed probe set drawn from all arms ("does this arm help everywhere?").
7. **Never tune ε, the step size or the arm grid on target accuracy** from the final run. Tune them during development, with development instruments as targets.

Optional second agent, only once steps 1–7 are done: Exp3, an adversarial bandit (Graves et al. use its non-stationary form, Exp3.S). In our toy test it learned more slowly than ε-greedy, so it is not the first choice.

**Output.** **Fig 10:** mean accuracy on unseen real instruments against the number of simulated chords, one line per condition with a ± std band. **Fig 11:** which arms the agent chose, early against late, as an α × blur heatmap with the real instruments' (α, blur) as dots. **Table 7:** final accuracy per condition (mean ± std over 20 runs), the simulated chords needed to reach uniform's final accuracy, and a Wilcoxon signed-rank test of ε-greedy against uniform, paired by home and seed (`scipy.stats.wilcoxon`).

### If time runs short, cut in this order

1. Exp3 2. Homes from 4 to 2 3. Seeds from 5 to 3 4. The confusion-match metric in E2

Never cut E1, E2 with its ATC baseline, or uniform against home only.

### What is known and what looks new

Already known; cite it:

- Overtones distort chroma and make some chords ambiguous. Peeters (2006) describes it; Papadopoulos & Peeters (2007) built chord templates that include harmonics; Mauch & Dixon (2010) model notes with decaying overtones, recover the notes with NNLS, and report gains of up to 12 points on harmonically difficult chords; Müller & Ewert (2010) designed chroma features that discard timbre.
- Predicting accuracy on shifted data from unlabelled examples: ATC (Garg et al. 2022).
- Bandits choosing what to train on, rewarded by learning progress: Graves et al. (2017). Learning which simulator settings to sample, instead of sampling uniformly: Active Domain Randomization (Mehta et al. 2019).

What looks new: a per-instrument 12-number fingerprint, fitted from 3 labelled chords, used to predict a trained chord classifier's accuracy and confusions on an unseen instrument and compared against ATC; and a bandit that chooses simulated fingerprints to train a chord classifier. Our fingerprint is close to the note model inside NNLS chroma, reduced to 12 numbers per instrument, so frame the contribution as the prediction and the curriculum, not the leakage model itself.

This rests on about ten web searches, not a literature review. Spend 2 hours on Google Scholar (try "chord recognition timbre robustness", "chroma overtone model instrument", "predict accuracy distribution shift audio", "curriculum bandit data augmentation audio") and ask Alec. The pilot numbers in the previous version of this guide came from synthetic audio; they have been removed and are not evidence.

---

## Stage 7: Presentation (Tue 1 Dec, 09:00, 8 minutes)

It is not graded, but it rehearses the story everyone will tell in their report.

| Time | Slide | Show | Play |
|---|---|---|---|
| 0:00–0:40 | 1. The task | A chord in, a name out | A keyboard clip, then the model's answer |
| 0:40–1:40 | 2. Data | 16 instruments, 7,398 chords, 7 types; Fig 2 | The same chord on 3 instruments |
| 1:40–2:30 | 3. Chroma | Fig 3: leakage onto B in C major | — |
| 2:30–3:40 | 4. Models and the fair test | Headline of Table 2; Fig 6 (seen, unseen, held-out) | — |
| 3:40–4:30 | 5. Where it fails | Fig 7a; the music theory behind the top confusion | 2 wrong clips |
| 4:30–5:30 | 6. Fingerprints | Fig 8; Fig 9 against ATC | A clip the fingerprint predicted the model would get wrong |
| 5:30–6:40 | 7. RL curriculum | Diagram of the bandit loop; Figs 10 and 11 | — |
| 6:40–7:30 | 8. Our keyboard, and limits | Table 4; what did not work | Our recording |
| 7:30–8:00 | 9. The answer | One sentence per hypothesis | — |

Preparation:

- Build slides only from the figures in `results/`.
- Embed the audio in the slides and test it on the laptop that will present. Keep a recording of the demo in case a live demo fails.
- Draft the slides by Sun 29 Nov. Do one timed rehearsal on Mon 30 Nov and cut until it runs under 7:45.
- Agree who presents which slides.
- Write down Alec's questions: they are free feedback for the reports.

---

## Stage 8: Individual reports (due Thu 10 Dec, 12:00)

Everyone writes alone, in their own words, from the shared results. The same figures and numbers across the group are expected; shared sentences are not. Confirm with Alec that shared figures are fine.

Aim for 1,850–1,900 words (the limit is 1,500–2,000). Use at most 6 or 7 figures and tables, each with a caption that states its takeaway.

| Section | Words | What goes in | Figures | Marking criteria it serves |
|---|---|---|---|---|
| Introduction | 250 | The task and why it is useful; why learning is needed (the baseline's mistakes); the research question and H1–H4; a one-sentence answer | — | Objectives clearly defined |
| Datasets | 350 | IDMT-SMT-Chords (source, 16 instruments, 7,398 chords, 7 types, inversions only on keyboards); the label table and how it was checked; class balance and why macro-F1; preprocessing (16 kHz, 2-s slices, attack trim and why); the splits (development and held-out, seen and unseen, repeats kept together); our keyboard recordings | Fig 1 or Table 1 | Methods clearly described; evaluation fair |
| Methods | 650 | Chroma and bass chroma from our own FFT (one equation); the template baseline; joint 84-class prediction and why; the 4 models and the CNN, one or two sentences each on how they work and what was tuned; grouped cross-validation; the fingerprint model and NNLS; the prediction method and ATC; the bandit (arms, reward, ε-greedy, why a constant step size) | Fig 3; a small diagram of the bandit loop | Methods clearly described; problems linked to ML concepts |
| Evaluation | 650 | Metrics; Table 2 with mean ± std; the seen–unseen gap and why it happens; the top confusions explained with music theory and listening; held-out and keyboard results; E1–E3 results against H1–H4; limitations; how the results relate to prior work | Table 2; Fig 6 or 7; Fig 9; Fig 10 | Evaluation fair and objective (objective metrics and subjective listening) |

**ML concepts to name where they genuinely apply:** generalisation and domain shift (seen against unseen); grouped cross-validation and leakage; class imbalance and macro-F1; linear separability (joint against separate prediction); bias, variance and regularisation (C, k, depth); the CNN's inductive bias (pooling over time, not pitch); confidence and calibration (ATC); exploration against exploitation and non-stationary bandits.

**Habits that score well:** every number with its uncertainty (mean ± std over folds or seeds); explain *why* each result happens, not only what it is; name the limitations (software instruments, one fingerprint per instrument, isolated 2-s chords, a small keyboard test set); compare with prior work (sources below); keep one research question throughout.

**Schedule:** outline by Thu 3 Dec (bullets and chosen figures for each section); full draft by Mon 7 Dec; submit Wed 9 Dec, a day early.

**Submission:**

- File names: `EXAM-NUMBER IAML Project Report.pdf` and `EXAM-NUMBER IAML Supp.zip`. No names anywhere, including code comments, file metadata and any git history inside the zip.
- The .zip holds `src/`, `infer.py`, the saved model in `models/`, `README.md`, `requirements.txt` and one short example .wav for `infer.py`. Leave out the dataset; the README says how to download it.
- Before submitting, unzip on a clean machine, install the requirements and run `python infer.py example.wav`.

---

## Sources

- IDMT-SMT-Chords dataset (Fraunhofer IDMT), Zenodo: https://zenodo.org/records/7544213
- Peeters (2006), Chroma-based estimation of musical key from audio-signal analysis, ISMIR: https://archives.ismir.net/ismir2006/paper/000134.pdf
- Papadopoulos & Peeters (2007), Large-scale study of chord estimation algorithms based on chroma representation and HMM, CBMI: https://hal.archives-ouvertes.fr/hal-00511437
- Mauch & Dixon (2010), Approximate note transcription for the improved identification of difficult chords, ISMIR: https://zenodo.org/record/1416598
- Müller & Ewert (2010), Towards timbre-invariant audio features for harmony-based music, IEEE TASLP: https://sebewert.github.io/publications_pdf/2010_MuellerEwert_MFCC-Chroma_IEEE-TASLP_postprint.pdf
- Garg et al. (2022), Leveraging unlabeled data to predict out-of-distribution performance (ATC), ICLR: https://arxiv.org/abs/2201.04234
- Graves et al. (2017), Automated curriculum learning for neural networks, ICML: https://arxiv.org/abs/1704.03003
- Mehta et al. (2019), Active Domain Randomization, CoRL: https://arxiv.org/abs/1904.04762
- Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., ch. 2 (bandits, non-stationary problems): http://incompleteideas.net/book/the-book-2nd.html
