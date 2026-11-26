import numpy as np, librosa
from features import features, SR, ATTACK_S

clips = np.load("data/clips_16k.npy").astype(np.float32)
for i in np.random.default_rng(0).choice(len(clips), 5, replace=False):
    ref = librosa.feature.chroma_stft(y=clips[i][int(ATTACK_S*SR):], sr=SR, n_fft=8192,
                                      hop_length=2048, tuning=0).mean(axis=1)
    print(i, np.corrcoef(features(clips[i])[:12], ref)[0, 1])