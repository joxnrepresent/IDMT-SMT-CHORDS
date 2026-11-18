"""
Models for the live demo (presentation.py).

Each model is one function marked with @Model("name"). Type that same name into a grid
cell in the demo and the cell shows that function's predictions. Saving this file
reloads it in the running demo.

Saving models from models.ipynb into src/models/:
    sklearn : joblib.dump(model, "models/svm.joblib")      (put the scaler in a Pipeline)
    torch   : torch.jit.script(model).save("models/cnn.pt")
    other   : joblib.dump(anything, "models/scaler.joblib")  (scalers, class names, ...)
"""
from pathlib import Path

import librosa
import numpy as np

SR = 16000
MODEL_DIR = Path(__file__).resolve().parent / "models"


class Model:
    """Registers a function under a grid name and turns its output into (label, confidence)."""
    registry = {}

    def __init__(self, name):
        self.name = name

    def __call__(self, fn):
        self.fn = fn
        Model.registry[self.name.strip().lower()] = self
        return fn

    def predict(self, clip):
        out = self.fn(clip)
        label, conf = out if isinstance(out, (tuple, list)) else (out, None)
        return str(label), (None if conf is None else float(conf))

    @staticmethod
    def find(name):
        return Model.registry.get(name.strip().lower())


_cache = {}


def load(filename):
    """Load a file from src/models/ once and reuse it (.pt -> TorchScript, else joblib)."""
    path = MODEL_DIR / filename
    if path not in _cache:
        if path.suffix in (".pt", ".pth"):
            import torch
            _cache[path] = torch.jit.load(str(path)).eval()
        else:
            import joblib
            _cache[path] = joblib.load(path)
    return _cache[path]


# ---------------------------------------------------------------------------
# TEMPLATE: copy this block once per model.
# ---------------------------------------------------------------------------

@Model("My model")                                    # the name you type into the grid
def my_model(clip):
    """
    clip   : np.ndarray, float32, 16 kHz mono, exactly 2 s (32000 samples), starting
             where the chord was struck (same as one row of clips_16k.npy).
    return : "C:maj7"            chord only (no confidence bar)
             ("C:maj7", 0.83)    chord + confidence 0..1
             Labels like "F#:min7", "A:7", "B:hdim7", "E:5" (dom7/m7b5/power also work).
             Two models (root + type)? Return f"{root}:{quality}", p_root * p_quality
    """
    model = load("my_model.joblib")                   # src/models/my_model.joblib

    # Features: must be computed exactly as in training.
    chroma = librosa.feature.chroma_stft(y=clip, sr=SR, n_fft=8192, hop_length=4096)
    features = np.concatenate([chroma.mean(axis=-1), chroma.std(axis=-1)])

    probs = model.predict_proba([features])[0]
    return model.classes_[probs.argmax()], probs.max()
