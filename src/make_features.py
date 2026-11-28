# src/make_features.py
import numpy as np
from features import features
clips = np.load("data/clips_16k.npy").astype(np.float32)
np.save("data/features.npy", np.array([features(c) for c in clips]))