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


ROOTS = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
         "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
QUALITIES = {"maj":[0,4,7], "min":[0,3,7], "maj7":[0,4,7,11], "min7":[0,3,7,10],
             "dom7":[0,4,7,10], "m7b5":[0,3,6,10], "power":[0,7]}      # keys must match labels.csv
PC = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]