# Stage 6 guide: chord project extension

This file hands over a conversation about Stage 6 (the extension) of Joan's chord recognition project. If you are the Claude reading it, your job is to be Joan's **guiding coding assistant**: help Joan explore hypotheses H1–H4, one step at a time, explaining things before doing them.

---

## 1. Your role and how to work with Joan

### Who Joan is

- First-year BSc Artificial Intelligence and Computer Science student at the University of Edinburgh.
- Strong self-taught Python and NumPy. Has built neural networks from scratch and done some reinforcement learning.
- Newer to professional research workflows, Git, and audio signal processing. **Not a professional.** Explain ideas before using them.
- Prefers clean, spacious layouts over dense walls of text.

### The protocol for every step

Joan asked for this structure every time:

1. **Explain what we're doing and why.** Use plain words first and jargon second. Give a small concrete example where possible, like "the B box for C major = (h[11] + h[7] + h[4]) / 3".
2. **Offer options before deciding.** Lay out 2–4 possible approaches with their trade-offs (time, risk, what each would show). Say which you'd recommend and why, then **let Joan choose**.
3. **Check novelty.** Search the current literature (Google Scholar, arXiv, ISMIR proceedings, DCASE) for work close to this step. Say honestly whether it looks new, partly new or already done, and cite what you found. Joan wants **genuine, unique novelty**, so if something has been done, suggest a twist that would make it new.
4. **Point out interesting branches**: side questions or extensions worth exploring, each with a rough cost in time.
5. **Explain the most effective way to decide.** For any decision (yours or Joan's), say what evidence would settle it, and prefer cheap tests that decide early.

### Working rules

- **One step at a time.** Don't jump ahead. Confirm Joan understands before moving on.
- **Explain, then do.** Before writing code, describe what it will do. Keep code short, and comment the non-obvious lines.
- **Use the roadmap's code as the starting point** (`chord project roadmap.md`, Stage 6, roughly lines 402–703). Change it only with a reason.
- **Respect the gates.** Each experiment depends on the previous one. If a gate fails, stop and treat the failure as a result ("a failure is still a result if we can say why").
- **Data hygiene.** Develop and debug on the 13 development instruments only. Run the final E2 and E3 on all 16 only after code and settings are frozen. Never tune settings using final target accuracy.
- **Commit the hypotheses to the repo, dated, before running E2 and E3**, so it's clear they were predictions and not explanations written afterwards.
- **Repeats.** Use 5 seeds or draws and report mean ± std.

---

## 2. Project context

- **Course:** IAML (Introductory Applied Machine Learning), group project. Joan is **Coder 1** and owns data, features, models and the extension.
- **Task:** hear 2 seconds of audio and name the chord: root C–B × 7 chord types = **84 classes**.
- **Dataset:** IDMT-SMT-Chords. 16 instruments (guitars, pianos, organ, pads; synthesised in Ableton Live and GarageBand), 7,398 clips.
- **Tools:** Python, scikit-learn, one small PyTorch CNN.
- **Project folder (on Joan's Mac):** `/Users/joanmk/UniStuff/IAML/IDMT-SMT-CHORDS`
  - `chord project roadmap.md`: the full plan (Stages 1–8)
  - `src/data/processed/clips_16k.npy` (clips at 16 kHz) and `src/data/processed/labels.csv`
  - `src/models.ipynb`, `src/extract_data.ipynb`, `src/presentation.py`, and others

### Deadlines

| What | When |
|---|---|
| E1: fit the fingerprints | Fri 20 Nov 2026 |
| E2: predict the mistakes | Mon 23 Nov 2026 |
| E3: bandit over simulated instruments | Fri 27 Nov 2026 |
| Group presentation (8 min, not graded) | Tue 1 Dec 2026, 09:00 |
| Individual report (1,500–2,000 words) + code .zip | Thu 10 Dec 2026, 12:00 |

### Known issues before starting (as of 17 Nov)

- **Chroma features may not exist yet.** All of Stage 6 needs the 12 chroma numbers per clip (`features.npy` from Stage 3). As of 17 Nov there was no `features.npy` in `src/`. Check first, and if it's missing, build it (roadmap Stage 3, `src/features.py` and `src/make_features.py`).
- **Paths differ.** The roadmap code loads `data/labels.csv` and `data/features.npy`, but Joan's labels live at `src/data/processed/labels.csv`. Fix the paths.
- **Check normalisation.** The fingerprint code assumes each chroma row sums to 1. Confirm how Stage 3 scales chroma.

---

## 3. The hypotheses

- **H1.** One fingerprint per instrument explains most of its chroma (high R²), and brighter instruments have larger *h*[7], *h*[4] and *h*[10].
- **H2.** Fingerprint predictions of accuracy on a new instrument are closer to the truth than ATC, the standard confidence-based method, and than testing on the 3 calibration chords.
- **H3.** Adding simulated instruments, chosen uniformly at random, raises accuracy on unseen real instruments over training on the home instrument alone.
- **H4.** The bandit reaches uniform sampling's final accuracy with fewer simulated chords, because it spends its budget on bright or blurred instruments early.

H1 goes with E1, H2 with E2, and H3 and H4 with E3.

---

## 4. The concepts, as Joan now understands them

### Chroma

The classifier never sees raw audio. It sees **chroma**: 12 numbers giving the energy on each note name (C, C♯ … B). Making chroma takes three steps:

1. Compute the spectrum.
2. Assign each frequency to its nearest semitone.
3. Fold the octaves together, so every C goes in one box.

The octave information is thrown away on purpose. Joan's question was "why not model the continuous spectrum?" The answer:

- The model only ever sees the 12 numbers, so we model the leakage *it* sees.
- Low overtones land almost exactly on semitones. Harmonic 3 is 2 cents off a fifth and harmonic 5 is 14 cents flat of a major third. Harmonic 7 is 31 cents off, so its energy splits between two boxes, which is part of what "blur" captures.
- Chord names depend only on note names.

### Leakage (overtones)

A real note also produces quieter overtones. Harmonics 3 and 6 land on the fifth, 5 and 10 on the major third, and 7 near the minor seventh. So every note **leaks** energy into other boxes. Example: in C major (C E G), both E's fifth and G's major third land on B, so a bright C major can look like **Cmaj7**.

**Key nuance:** leakage alone doesn't break the model. Trained and tested on the same instrument, it learns that instrument's leakage. The problem is **mismatch**: a model trained on a dull instrument and tested on a bright one sees unfamiliar leakage. That's why every experiment uses "train on home instrument, test on target instrument".

### The fingerprint *h*

**12 numbers: the chroma of a single note on that instrument, written relative to that note.**

- h[0] is the note and its octaves, h[7] a fifth up, h[4] a major third up, h[10] a minor seventh up.
- A pure sine tone would be `[1, 0, …, 0]`.

It rests on two assumptions:

1. Every note has the same pattern, just shifted.
2. Notes in a chord simply add, so chord chroma = the average of *h* shifted to each note.

### Fitting the fingerprint (regression)

The dataset only has chords, so *h* is fitted backwards:

- **Known:** each clip's chord notes (from the label) and its measured chroma.
- **Unknown:** the 12 values of *h* (they act as the regression weights).
- Each of the 12 boxes in each clip gives one linear equation. For C major, the B box = (h[11] + h[7] + h[4]) / 3.
- Roughly 460 clips × 12 equations per instrument, against only 12 unknowns.
- Solve with **NNLS** (non-negative least squares: line-fitting where no weight may be negative, because energy can't be negative), then scale *h* to sum to 1.
- **R²** measures how much of the real chroma the fitted *h* explains.

In the roadmap code, `mix_matrix` builds the equations for one clip, `np.vstack` stacks all clips, and `nnls` solves.

**If the fit is bad,** one fingerprint can't describe that instrument. Likely reasons are a tone that changes with register, reverb carrying over, or distortion. That's a finding to report, and it means E2's predictions for that instrument are less trustworthy.

### E1: fit the fingerprints

Fit *h* and R² for all 16 instruments. Check that bright instruments have bigger h[7] and h[4]. Outputs: Fig 8 (a heatmap of the fingerprints) and Table 5 (R² per instrument plus fitted α and blur).

### E2: predict the mistakes

Real-world setting: the model was trained on piano, someone brings a guitar, and you want to know how well the model will do **without labelling hundreds of guitar clips**.

1. Take 3 labelled clips of the target instrument.
2. Fit its fingerprint from them. 3 clips give 36 numbers, enough for 12 unknowns but too few to test on directly.
3. Simulate the chroma of all 84 chords (adding the home instrument's residual noise).
4. The model's accuracy on the simulated chords is the prediction.

Compare against the **actual** accuracy, against **ATC** (Garg et al. 2022, which predicts accuracy from model confidence), and against plain accuracy on the 3 clips. Run all 240 home/target pairs × 5 draws.

**E2 also tests whether the simulator is realistic.** If predictions match reality, simulated chroma behaves like real chroma, which justifies training on simulated data in E3.

### E3: simulated instruments and a bandit

**Generator.** Two knobs make a fingerprint:

- **α (brightness):** harmonic *k* has amplitude *k*^−α, so a low α is bright.
- **blur:** spreads energy onto neighbouring semitones.

Fingerprint → chroma for any chord → labelled training data for an instrument that doesn't exist.

**Uniform random (H3)** is domain randomisation. Train on the home instrument plus many varied fake ones, so the model learns chord identity regardless of brightness. It does **not** look at where the model fails on real instruments; it never sees them.

**Bandit (H4).** There are 21 knob settings (the "arms"). ε-greedy picks which setting to generate next. The reward is **learning progress**: the drop in log-loss on a fresh probe batch *from the same arm*. Two details:

- It never uses real target instruments, which would be cheating.
- It rewards *improvement*, not error, so it doesn't fixate on settings no model can learn (Graves et al. 2017).

**Gates, in order:**

1. Toy-bandit test.
2. Plot simulated against real chroma by eye.
3. Uniform against home only. **If uniform doesn't help, stop and report H3 as rejected.**
4. Fixed-arm sweep.
5. Then the bandit.

There's also a go/no-go after E2: if nearly every pair is above about 97% accuracy, run E3 only on the hardest home instruments.

**If time runs short, cut in this order:** Exp3, then homes 4→2, then seeds 5→3, then the E2 confusion-match metric. **Never cut** E1, E2 with ATC, or uniform against home only.

---

## 5. Ideas and branches raised so far

1. **Fingerprint adaptation (a gap in the roadmap).** Joan asked: "why not just generate chords for the new instrument and add them to training?" Fit *h* from the 3 target clips, simulate all 84 chords, retrain, and measure the gain. It's simple, it might be strong, and it could be an extra baseline or experiment. It differs from E3 because it uses 3 labelled target clips, while E3 uses none.
2. **"How many numbers does an instrument need?"** Compare model complexity on **held-out clips** (fit on half of each instrument's clips, test on the other half): 2 numbers (α, blur) → 12 (fingerprint) → richer models. Held-out testing matters because more parameters always fit the training data better.
3. **Richer leakage models**, worth trying only if R² is low. Options are per-register fingerprints (these need a feature that keeps octaves, such as an 84-bin pitch spectrum instead of chroma) or a neural network mapping chord notes to chroma. Reasons to stay at 12 numbers by default: E2 must fit from only 3 clips, h[k] is readable (which H1 needs), E3 needs a few simple knobs, and time is short. **Let the data decide**: high R² means little to gain, and a low one means looking at the residuals and adding exactly what's missing.

---

## 6. Novelty: what's known so far

This is from the roadmap and is based on **only about 10 web searches, not a literature review**. Do a proper check at each step.

**Already known (cite it):**

- Overtones distort chroma: Peeters (2006); Papadopoulos & Peeters (2007), who built chord templates with harmonics; Mauch & Dixon (2010), whose NNLS chroma uses a note model with decaying overtones; Müller & Ewert (2010), who designed timbre-robust chroma.
- Predicting accuracy on shifted data from unlabelled examples: ATC (Garg et al. 2022).
- Bandits choosing training data, rewarded by learning progress: Graves et al. (2017). Learning which simulator settings to sample: Active Domain Randomization (Mehta et al. 2019).

**What looks new:**

- A per-instrument 12-number fingerprint, fitted from 3 labelled chords, used to predict a chord classifier's accuracy **and confusions** on an unseen instrument, compared against ATC.
- A bandit choosing simulated fingerprints to train a chord classifier.
- Frame the contribution as **the prediction and the curriculum, not the leakage model itself**, which is close to the note model inside NNLS chroma.

**Suggested searches:** "chord recognition timbre robustness", "chroma overtone model instrument", "predict accuracy distribution shift audio", "curriculum bandit data augmentation audio", "few-shot instrument adaptation chord". Also ask Alec, presumably the tutor or supervisor.

---

## 7. Where we left off

Joan understands the concepts in section 4 and wants to tackle the hypotheses **one by one, starting with E1 / H1**.

**Suggested first steps** (still follow the protocol: explain, give options, let Joan choose):

1. **Step 0:** check whether chroma features exist. If not, build them from `clips_16k.npy` (Stage 3).
2. **Step 1:** commit H1–H4 to the repo with today's date.
3. **Step 2:** create `src/fingerprint.py` with the E1 functions, fixing the paths.
4. **Step 3:** fit *h* and R² on the 13 development instruments, then all 16. Make Fig 8 and Table 5.
5. **Step 4:** sanity-check H1. Do brighter instruments have larger h[7], h[4] and h[10]? Then decide, with Joan, whether to move on to E2.
