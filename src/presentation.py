"""
Live chord-recognition demo.

    .venv/bin/python src/presentation.py

Keys:  Space  live / strum      S  settings      K  open / close synth drawer
       F      fullscreen        R  reload presentation_models.py (also automatic on save)

Grid  : click "+" and type a model name -> the cell shows the @Model("name") function from
        presentation_models.py. Double-click a cell to rename it, hover + x to remove it.
Live  : every 0.25 s the models see the last 2 s of audio (never reaching back past
        the latest chord change; short windows are looped up to 2 s).
Strum : each new chord is recorded for exactly 2 s from its onset, then predicted once.
Synth : drag or click the handle to lift / lower it. Right-click a control to assign a MIDI CC.
"""
import importlib
import json
import math
import signal
import sys
import threading
import time
import traceback
from collections import deque
from fractions import Fraction
from pathlib import Path

import mido
import numpy as np
import sounddevice as sd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from scipy.signal import lfilter, resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MODELS_FILE = HERE / "presentation_models.py"
pm = None      # the presentation_models module (loaded by import_models)

CONFIG_PATH = HERE / "presentation_config.json"
SR = 16000
CLIP_S = 2.0
LIVE_HOP_S = 0.25
MIN_LIVE_S = 0.4      # shortest window (after a chord change) worth predicting on
PREROLL_S = 0.02      # start clips slightly before the detected onset
mido.set_backend("mido.backends.rtmidi")

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG = QtGui.QColor("#0b0b0d")
CELL = QtGui.QColor("#141417")
CELL_HOVER = QtGui.QColor("#18181c")
TEXT = QtGui.QColor("#f2f2f5")
DIM = QtGui.QColor("#6e6e78")
FAINT = QtGui.QColor("#26262c")
ACCENT = QtGui.QColor("#8fa8ff")
GOOD = QtGui.QColor("#5fd38d")
BAD = QtGui.QColor("#e5646b")


def with_alpha(c, a):
    c = QtGui.QColor(c)
    c.setAlphaF(max(0.0, min(1.0, a)))
    return c


# ---------------------------------------------------------------------------
# Chord names
# ---------------------------------------------------------------------------
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = {"maj": [0, 4, 7], "min": [0, 3, 7], "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10],
             "7": [0, 4, 7, 10], "hdim7": [0, 3, 6, 10], "5": [0, 7]}
NOTE_TO_PC = {n: i for i, n in enumerate(NOTES)}
NOTE_TO_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0})
QUALITY_ALIASES = {"maj": "maj", "": "maj", "min": "min", "m": "min", "maj7": "maj7", "min7": "min7",
                   "m7": "min7", "7": "7", "dom7": "7", "hdim7": "hdim7", "m7b5": "hdim7",
                   "5": "5", "power": "5"}
QUALITY_DISPLAY = {"maj": "maj", "min": "min", "maj7": "maj7", "min7": "min7", "7": "7",
                   "hdim7": "m7♭5", "5": "5"}


def parse_label(label):
    """'C#:maj7/3' -> (1, 'maj7'); 'C' -> (0, None); unknown -> None."""
    s = str(label).strip().split("/")[0]
    root, sep, q = s.partition(":")
    pc = NOTE_TO_PC.get(root.strip())
    if pc is None:
        return None
    if not sep:
        return pc, None
    return pc, QUALITY_ALIASES.get(q.strip(), q.strip())


def pretty(label):
    """-> (root text, quality text)."""
    p = parse_label(label)
    if p is None:
        return str(label), ""
    pc, q = p
    root = NOTES[pc].replace("#", "♯")
    return root, ("" if q is None else QUALITY_DISPLAY.get(q, q))


def same_chord(pred, truth):
    a, b = parse_label(pred), parse_label(truth)
    if a is None or b is None:
        return None
    if a[1] is None:
        return a[0] == b[0]
    return a == b


def identify_chord(notes):
    """MIDI note numbers -> 'C:maj7' (or note names if not in the 84-chord vocabulary)."""
    if not notes:
        return None
    pcs = {n % 12 for n in notes}
    bass = min(notes) % 12
    for root in [bass] + [r for r in range(12) if r != bass]:
        for q, ivs in QUALITIES.items():
            if {(root + i) % 12 for i in ivs} == pcs:
                return f"{NOTES[root]}:{q}"
    return None


def note_names(notes):
    return " ".join(NOTES[n % 12].replace("#", "♯") for n in sorted(notes))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KNOBS = [  # key, label, default
    ("cutoff", "Cutoff", 0.75),
    ("res", "Res", 0.2),
    ("attack", "Attack", 0.1),
    ("decay", "Decay", 0.5),
    ("sustain", "Sustain", 0.8),
    ("release", "Release", 0.35),
    ("volume", "Volume", 0.6),
]
WAVES = ["saw", "square", "triangle", "sine"]   # knobs["wave"] in 0..1 picks one of four


def knob_value(key, v):
    """Normalised 0..1 knob -> physical value."""
    if key == "wave":
        return WAVES[min(3, int(v * 4))]
    if key == "cutoff":
        return 60.0 * (16000 / 60.0) ** v
    if key == "res":
        return 0.5 * 20.0 ** v
    if key == "attack":
        return 0.002 * 1000.0 ** v
    if key == "decay":
        return 0.01 * 300.0 ** v
    if key == "release":
        return 0.01 * 400.0 ** v
    return v


def knob_text(key, v):
    x = knob_value(key, v)
    if key == "wave":
        return x
    if key == "cutoff":
        return f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"
    if key == "res":
        return f"{x:.1f}"
    if key in ("attack", "decay", "release"):
        return f"{x * 1000:.0f}ms" if x < 1 else f"{x:.1f}s"
    return f"{round(x * 100)}%"


DEFAULT_CFG = {
    "source": "midi",          # "midi" | "audio"
    "mode": "live",            # "live" | "strum"
    "audio_in": None,          # device name (None = system default)
    "channel": 0,              # 0 = mix all channels, n = channel n
    "audio_out": None,
    "midi_in": None,           # port name (None = first available)
    "gate_db": -50,
    "models": [],              # names typed into the grid
    "knobs": {"wave": 0.0, **{k: d for k, _, d in KNOBS}},
    "cc_map": {},              # "74": "cutoff"
    "synth_open": True,
}


def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    try:
        saved = json.loads(CONFIG_PATH.read_text())
        knobs = {**cfg["knobs"], **saved.get("knobs", {})}
        cfg.update(saved)
        cfg["knobs"] = knobs
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Synth
# ---------------------------------------------------------------------------
def poly_blep(t, dt):
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt
    out[m] = x + x - x * x - 1.0
    m = t > 1.0 - dt
    x = (t[m] - 1.0) / dt
    out[m] = x * x + x + x + 1.0
    return out


class Envelope:
    def __init__(self):
        self.stage, self.level, self.rel_from = "off", 0.0, 0.0

    def on(self):
        self.stage = "a"

    def off(self):
        if self.stage != "off":
            self.stage, self.rel_from = "r", max(self.level, 1e-6)

    def render(self, n, a, d, s, r, sr):
        out = np.zeros(n, np.float32)
        i, lvl = 0, self.level
        while i < n:
            st = self.stage
            if st == "a":
                rate = 1.0 / (a * sr)
                k = min(n - i, max(1, math.ceil((1.0 - lvl) / rate)))
                seg = np.minimum(lvl + rate * np.arange(1, k + 1), 1.0)
                out[i:i + k], lvl, i = seg, float(seg[-1]), i + k
                if lvl >= 1.0:
                    self.stage = "d"
            elif st == "d":
                rate = (1.0 - s) / (d * sr)
                if lvl <= s or rate <= 0:
                    self.stage = "s"
                    continue
                k = min(n - i, max(1, math.ceil((lvl - s) / rate)))
                seg = np.maximum(lvl - rate * np.arange(1, k + 1), s)
                out[i:i + k], lvl, i = seg, float(seg[-1]), i + k
                if lvl <= s:
                    self.stage = "s"
            elif st == "s":
                lvl = s
                out[i:] = s
                i = n
            elif st == "r":
                rate = self.rel_from / (r * sr)
                k = min(n - i, max(1, math.ceil(lvl / rate)))
                seg = np.maximum(lvl - rate * np.arange(1, k + 1), 0.0)
                out[i:i + k], lvl, i = seg, float(seg[-1]), i + k
                if lvl <= 0.0:
                    self.stage = "off"
            else:
                i = n
        self.level = lvl
        return out


class Voice:
    def __init__(self, note, vel):
        self.note, self.freq = note, 440.0 * 2 ** ((note - 69) / 12)
        self.phase, self.amp = 0.0, 0.22 * (0.4 + 0.6 * vel / 127)
        self.env, self.held = Envelope(), True
        self.env.on()


class Synth:
    MAX_VOICES = 24

    def __init__(self, sr, knobs):
        self.sr, self.knobs = sr, knobs
        self.voices = {}
        self.events = deque()
        self.pedal = False
        self.zi = np.zeros(2)
        self.cutoff = None
        self.sounding = frozenset()    # notes not yet released (for the "true chord")

    # called from the MIDI thread
    def note_on(self, note, vel):
        self.events.append(("on", note, vel))

    def note_off(self, note):
        self.events.append(("off", note, 0))

    def set_pedal(self, down):
        self.events.append(("pedal", down, 0))

    def _handle_events(self):
        n_on = 0
        while self.events:
            kind, a, b = self.events.popleft()
            if kind == "on" and b > 0:
                n_on += 1
                v = self.voices.get(a)
                if v is None:
                    if len(self.voices) >= self.MAX_VOICES:
                        oldest = min(self.voices.values(), key=lambda x: (x.held, x.env.level))
                        del self.voices[oldest.note]
                    self.voices[a] = Voice(a, b)
                else:
                    v.held, v.amp = True, 0.22 * (0.4 + 0.6 * b / 127)
                    v.env.on()
            elif kind in ("on", "off"):
                v = self.voices.get(a)
                if v is not None:
                    v.held = False
                    if not self.pedal:
                        v.env.off()
            elif kind == "pedal":
                self.pedal = a
                if not a:
                    for v in self.voices.values():
                        if not v.held:
                            v.env.off()
        return n_on

    def render(self, n):
        """-> (audio for the models, number of note-ons in this block)."""
        n_on = self._handle_events()
        k = self.knobs
        wave = knob_value("wave", k["wave"])
        a, d = knob_value("attack", k["attack"]), knob_value("decay", k["decay"])
        s, r = k["sustain"], knob_value("release", k["release"])
        mix = np.zeros(n, np.float64)
        ramp = np.arange(1, n + 1)
        for note in list(self.voices):
            v = self.voices[note]
            env = v.env.render(n, a, d, s, r, self.sr)
            if v.env.stage == "off":
                del self.voices[note]
                continue
            dt = v.freq / self.sr
            ph = (v.phase + dt * ramp) % 1.0
            v.phase = float(ph[-1])
            if wave == "saw":
                w = 2.0 * ph - 1.0 - poly_blep(ph, dt)
            elif wave == "square":
                w = np.where(ph < 0.5, 1.0, -1.0) + poly_blep(ph, dt) - poly_blep((ph + 0.5) % 1.0, dt)
            elif wave == "triangle":
                w = 1.0 - 4.0 * np.abs(ph - 0.5)
            else:
                w = np.sin(2 * np.pi * ph)
            mix += w * env * v.amp
        self.sounding = frozenset(v.note for v in self.voices.values() if v.env.stage in "ads")

        # resonant low-pass (RBJ biquad), cutoff smoothed per block
        target = min(knob_value("cutoff", k["cutoff"]), 0.45 * self.sr)
        self.cutoff = target if self.cutoff is None else self.cutoff + 0.3 * (target - self.cutoff)
        q = knob_value("res", k["res"])
        w0 = 2 * np.pi * self.cutoff / self.sr
        alpha, cw = np.sin(w0) / (2 * q), np.cos(w0)
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
        a_ = np.array([1 + alpha, -2 * cw, 1 - alpha])
        y, self.zi = lfilter(b / a_[0], a_ / a_[0], mix, zi=self.zi)
        return np.tanh(y).astype(np.float32), n_on


# ---------------------------------------------------------------------------
# Audio engine
# ---------------------------------------------------------------------------
class Ring:
    """Rolling audio buffer addressed by absolute sample index."""

    def __init__(self, sr, seconds=10):
        self.sr, self.size = sr, int(sr * seconds)
        self.buf = np.zeros(self.size, np.float32)
        self.count = 0
        self.lock = threading.Lock()

    def write(self, x):
        n = len(x)
        with self.lock:
            i = self.count % self.size
            j = min(n, self.size - i)
            self.buf[i:i + j] = x[:j]
            self.buf[:n - j] = x[j:]
            self.count += n

    def read(self, start, end):
        with self.lock:
            start, end = max(int(start), self.count - self.size, 0), min(int(end), self.count)
            if end <= start:
                return np.zeros(0, np.float32)
            idx = np.arange(start, end) % self.size
            return self.buf[idx].copy()


def find_device(name, kind):
    if name:
        for i, dev in enumerate(sd.query_devices()):
            if dev["name"] == name and dev[f"max_{kind}_channels"] > 0:
                return i
    return None


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.knobs = cfg["knobs"]
        self.stream = self.midi = self.ring = self.synth = None
        self.onset = None           # absolute sample index of the latest chord onset
        self.learn = None           # knob key waiting for a CC
        self.status = ""
        self.midi_name = None
        self._slow = 0.0
        self._last_onset = -10 ** 9

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self.stop()
        self.status, self.onset = "", None
        self._open_midi()
        try:
            if self.cfg["source"] == "midi":
                dev = find_device(self.cfg["audio_out"], "output")
                sr = int(sd.query_devices(dev, "output")["default_samplerate"])
                self.ring, self.synth = Ring(sr), Synth(sr, self.knobs)
                self.stream = sd.OutputStream(device=dev, samplerate=sr, channels=1, blocksize=512,
                                              dtype="float32", latency="low", callback=self._out_cb)
            else:
                dev = find_device(self.cfg["audio_in"], "input")
                info = sd.query_devices(dev, "input")
                sr = int(info["default_samplerate"])
                self.ring, self.synth = Ring(sr), None
                self.stream = sd.InputStream(device=dev, samplerate=sr, channels=info["max_input_channels"],
                                             blocksize=512, dtype="float32", callback=self._in_cb)
            self.stream.start()
        except Exception as e:  # noqa: BLE001
            self.status = f"audio: {e}"
            self.stream = None

    def stop(self):
        for s in (self.stream, self.midi):
            try:
                if s is not None:
                    s.close()
            except Exception:  # noqa: BLE001
                pass
        self.stream = self.midi = None

    def _open_midi(self):
        names = mido.get_input_names()
        name = self.cfg["midi_in"] if self.cfg["midi_in"] in names else (names[0] if names else None)
        self.midi_name = name
        if name:
            try:
                self.midi = mido.open_input(name, callback=self._midi_cb)
            except Exception as e:  # noqa: BLE001
                self.status = f"midi: {e}"

    # -- callbacks (audio / MIDI threads) -----------------------------------
    def _out_cb(self, out, frames, t, status):
        try:
            x, n_on = self.synth.render(frames)
            if n_on:
                self._mark_onset(self.ring.count)
            vol = self.knobs["volume"]
            out[:, 0] = x * vol
            self.ring.write(x)
        except Exception:  # noqa: BLE001
            out.fill(0)
            traceback.print_exc()

    def _in_cb(self, indata, frames, t, status):
        ch = self.cfg["channel"]
        x = indata[:, ch - 1] if 0 < ch <= indata.shape[1] else indata.mean(axis=1)
        x = np.ascontiguousarray(x, np.float32)
        rms = float(np.sqrt(np.mean(x * x))) + 1e-9
        gate = 10 ** (self.cfg["gate_db"] / 20)
        if rms > gate and rms > 2.0 * self._slow and self.ring.count - self._last_onset > 0.3 * self.ring.sr:
            self._mark_onset(self.ring.count)
        self._slow = 0.9 * self._slow + 0.1 * rms
        self.ring.write(x)

    def _mark_onset(self, idx):
        self._last_onset = idx
        self.onset = idx

    def _midi_cb(self, msg):
        if msg.type == "control_change":
            key = str(msg.control)
            if self.learn:
                cc = self.cfg["cc_map"]
                for c in [c for c, k in cc.items() if k == self.learn]:
                    del cc[c]
                cc[key] = self.learn
                self.learn = None
            elif msg.control == 64 and self.synth:
                self.synth.set_pedal(msg.value >= 64)
            elif key in self.cfg["cc_map"]:
                self.knobs[self.cfg["cc_map"][key]] = msg.value / 127
        elif self.synth is not None and msg.type == "note_on":
            self.synth.note_on(msg.note, msg.velocity)
        elif self.synth is not None and msg.type == "note_off":
            self.synth.note_off(msg.note)

    # -- used by the inference thread ----------------------------------------
    def true_chord(self):
        if self.synth is None or not self.synth.sounding:
            return None
        return identify_chord(self.synth.sounding) or note_names(self.synth.sounding)


def to_clip(x, sr):
    """Device-rate audio -> exactly 2 s at 16 kHz (short windows are looped after the attack)."""
    if sr != SR:
        f = Fraction(SR, sr).limit_denominator(1000)
        x = resample_poly(x, f.numerator, f.denominator).astype(np.float32)
    n = int(CLIP_S * SR)
    if len(x) >= n:
        return np.ascontiguousarray(x[-n:], np.float32)
    body = x[int(0.1 * SR):] if len(x) > int(0.3 * SR) else x
    reps = math.ceil((n - len(x)) / max(1, len(body)))
    return np.concatenate([x, np.tile(body, reps)])[:n].astype(np.float32)


# ---------------------------------------------------------------------------
# Models + inference thread
# ---------------------------------------------------------------------------
def import_models():
    """(Re)load presentation_models.py. Returns an error message, or '' on success."""
    global pm
    try:
        pm = importlib.reload(pm) if pm else importlib.import_module("presentation_models")
        return ""
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return f"presentation_models.py: {type(e).__name__}: {e}"


class Slot:
    """One grid cell: a typed name, bound to the @Model with that name (if any)."""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.state, self.message = "ok", ""
        self.label = self.conf = self.truth = None
        self.changed = 0.0

    def model(self):
        reg = getattr(pm, "Model", None)
        return reg.find(self.name) if reg else None


class Inference(threading.Thread):
    def __init__(self, engine, slots):
        super().__init__(daemon=True)
        self.engine, self.slots = engine, slots
        self.running = True
        self.silent = True
        self.progress = None     # strum capture progress 0..1
        self.level_db = -100.0
        self._last = 0.0
        self._done_onset = None

    def run(self):
        while self.running:
            time.sleep(0.02)
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                time.sleep(0.5)

    def tick(self):
        eng = self.engine
        ring = eng.ring
        if ring is None or eng.stream is None:
            self.silent, self.progress = True, None
            return
        sr, now = ring.sr, ring.count
        recent = ring.read(now - 0.3 * sr, now)
        rms = float(np.sqrt(np.mean(recent ** 2))) if len(recent) else 0.0
        self.level_db = 20 * math.log10(rms + 1e-9)
        active = self.level_db > eng.cfg["gate_db"]
        onset = eng.onset
        pre = int(PREROLL_S * sr)

        if eng.cfg["mode"] == "live":
            self.progress = None
            if time.monotonic() - self._last < LIVE_HOP_S:
                return
            self.silent = not active
            if not active:
                return
            start = now - int(CLIP_S * sr)
            if onset is not None and onset - pre > start:
                start = onset - pre
            if now - start < MIN_LIVE_S * sr:
                return
            self._last = time.monotonic()
            self.predict(to_clip(ring.read(start, now), sr))
        else:
            self.silent = False
            if onset is None or onset == self._done_onset:
                self.progress = None
                return
            end = onset - pre + int(CLIP_S * sr)
            self.progress = min(1.0, (now - onset) / (CLIP_S * sr))
            if now >= end:
                self._done_onset = onset
                self.progress = None
                self.predict(to_clip(ring.read(onset - pre, end), sr))

    def predict(self, clip):
        truth = self.engine.true_chord()
        for s in list(self.slots):
            model = s.model()
            if model is None:
                continue
            try:
                label, conf = model.predict(clip)
                if label != s.label:
                    s.changed = time.monotonic()
                s.label, s.conf, s.truth, s.state = label, conf, truth, "ok"
            except FileNotFoundError as e:
                s.state, s.message = "missing", str(e)
            except Exception as e:  # noqa: BLE001
                if s.state != "error":
                    traceback.print_exc()
                s.state, s.message = "error", f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
def font(px, weight=QtGui.QFont.Weight.Normal):
    f = QtGui.QFont()
    f.setPixelSize(max(1, int(px)))
    f.setWeight(weight)
    return f


class InlineEdit(QtWidgets.QLineEdit):
    """Small text box for typing a model name. Enter/click away = done, Esc = cancel."""

    def __init__(self, parent, text, names, done, placeholder=""):
        super().__init__(text, parent)
        self.done, self.finished = done, False
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("QLineEdit{background:#0f0f12;color:#f2f2f5;border:1px solid #33333a;"
                           "border-radius:8px;padding:4px 10px;font-size:15px}")
        comp = QtWidgets.QCompleter(names, self)
        comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setCompleter(comp)
        self.returnPressed.connect(lambda: self.finish(self.text().strip()))
        self.selectAll()
        self.show()
        self.setFocus()

    def finish(self, text):
        if self.finished:
            return
        self.finished = True
        self.hide()
        self.deleteLater()
        self.done(text)

    def event(self, e):
        if e.type() == QtCore.QEvent.Type.ShortcutOverride and e.key() == Qt.Key.Key_Escape:
            e.accept()
            return True
        return super().event(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.finish("")
        else:
            super().keyPressEvent(e)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        if e.reason() != Qt.FocusReason.PopupFocusReason:
            self.finish(self.text().strip())


class ChordCell(QtWidgets.QWidget):
    removed = QtCore.Signal(object)
    renamed = QtCore.Signal(object)

    def __init__(self, slot, app):
        super().__init__()
        self.slot, self.app = slot, app
        self.setMouseTracking(True)
        self.hover = False

    def pad(self):
        return max(14, min(self.width(), self.height()) * 0.07)

    def close_rect(self):
        return QtCore.QRectF(self.width() - 34, 10, 24, 24)

    def enterEvent(self, e):
        self.hover = True
        self.update()

    def leaveEvent(self, e):
        self.hover = False
        self.update()

    def mousePressEvent(self, e):
        if self.hover and self.close_rect().contains(e.position()):
            self.removed.emit(self)

    def mouseDoubleClickEvent(self, e):
        pad = self.pad()
        ed = InlineEdit(self, self.slot.name, self.app.model_names(), self.finish_rename)
        ed.setGeometry(int(pad - 10), int(pad * 0.8 - 6), int(min(self.width() - 2 * pad, 280)), 34)

    def finish_rename(self, text):
        if text and text != self.slot.name:
            self.slot.name = text
            self.slot.reset()
            self.renamed.emit(self)

    def paintEvent(self, e):
        s, w, h = self.slot, self.width(), self.height()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(CELL_HOVER if self.hover else CELL)
        p.drawRoundedRect(QtCore.QRectF(0, 0, w, h), 14, 14)

        pad = self.pad()
        p.setFont(font(max(12, min(w, h) * 0.055)))
        p.setPen(DIM)
        p.drawText(QtCore.QRectF(pad, pad * 0.8, w - 2 * pad - 24, 30), Qt.AlignmentFlag.AlignLeft, s.name)
        if self.hover:
            p.setFont(font(18))
            p.drawText(self.close_rect(), Qt.AlignmentFlag.AlignCenter, "×")

        msg = None
        if s.model() is None:
            msg, col, tip = "no model", QtGui.QColor("#4a4a53"), f'No @Model("{s.name}") in presentation_models.py'
        elif s.state in ("missing", "error"):
            msg = "not trained" if s.state == "missing" else "error"
            col, tip = (DIM if s.state == "missing" else BAD), s.message
        if msg:
            p.setFont(font(max(12, min(w, h) * 0.06)))
            p.setPen(col)
            p.drawText(QtCore.QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, msg)
            self.setToolTip(tip)
            p.end()
            return
        self.setToolTip("")

        app = self.app
        dim = app.inference.silent or app.inference.progress is not None
        fade = min(1.0, (time.monotonic() - s.changed) / 0.15)
        alpha = (0.28 if dim else 1.0) * fade

        if s.label is None:
            p.setFont(font(min(h * 0.3, w * 0.2), QtGui.QFont.Weight.Light))
            p.setPen(FAINT)
            p.drawText(QtCore.QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "–")
        else:
            root, qual = pretty(s.label)
            big = min(h * 0.36, w * 0.26)
            fr, fq = font(big, QtGui.QFont.Weight.DemiBold), font(big * 0.42)
            mr, mq = QtGui.QFontMetricsF(fr), QtGui.QFontMetricsF(fq)
            gap = big * 0.06 if qual else 0
            total = mr.horizontalAdvance(root) + gap + mq.horizontalAdvance(qual)
            x = (w - total) / 2
            base = h / 2 + mr.capHeight() / 2
            p.setFont(fr)
            p.setPen(with_alpha(TEXT, alpha))
            p.drawText(QtCore.QPointF(x, base), root)
            if qual:
                p.setFont(fq)
                p.setPen(with_alpha(QtGui.QColor("#a9a9b4"), alpha))
                p.drawText(QtCore.QPointF(x + mr.horizontalAdvance(root) + gap, base), qual)

        # confidence bar
        bh = 4
        y = h - pad - bh
        bw = w - 2 * pad
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(FAINT)
        p.drawRoundedRect(QtCore.QRectF(pad, y, bw, bh), 2, 2)
        if s.label is not None and s.conf is not None:
            ok = same_chord(s.label, s.truth) if s.truth else None
            col = ACCENT if ok is None else (GOOD if ok else BAD)
            p.setBrush(with_alpha(col, 0.35 if dim else 1.0))
            p.drawRoundedRect(QtCore.QRectF(pad, y, max(bh, bw * s.conf), bh), 2, 2)
            p.setFont(font(max(11, min(w, h) * 0.045)))
            p.setPen(DIM)
            p.drawText(QtCore.QRectF(pad, y - 26, bw, 20),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, f"{s.conf * 100:.0f}%")
        p.end()


class AddCell(QtWidgets.QWidget):
    """Dashed '+' cell. Click it, type any model name, press Enter."""
    added = QtCore.Signal(str)

    def __init__(self, names):
        super().__init__()
        self.names = names
        self.edit = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hover = False

    def enterEvent(self, e):
        self.hover = True
        self.update()

    def leaveEvent(self, e):
        self.hover = False
        self.update()

    def mousePressEvent(self, e):
        if self.edit is None:
            w = min(self.width() - 40, 280)
            self.edit = InlineEdit(self, "", self.names(), self.done, placeholder="Model name")
            self.edit.setGeometry(int((self.width() - w) / 2), int(self.height() / 2 - 17), int(w), 34)
            self.update()

    def done(self, text):
        self.edit = None
        self.update()
        if text:
            self.added.emit(text)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(DIM if self.hover or self.edit else FAINT, 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QtCore.QRectF(1, 1, self.width() - 2, self.height() - 2), 14, 14)
        if self.edit is None:
            p.setFont(font(min(self.width(), self.height()) * 0.18, QtGui.QFont.Weight.Light))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "+")
        p.end()


class Grid(QtWidgets.QWidget):
    def __init__(self, names):
        super().__init__()
        self.cells = []
        self.add = AddCell(names)
        self.add.setParent(self)

    def set_cells(self, cells):
        for c in self.cells:
            if c not in cells:
                c.setParent(None)
                c.deleteLater()
        self.cells = cells
        for c in cells:
            c.setParent(self)
            c.show()
        self.relayout()

    def resizeEvent(self, e):
        self.relayout()

    def relayout(self):
        items = self.cells + [self.add]
        n, W, H, gap, m = len(items), self.width(), self.height(), 14, 18
        best = None
        for cols in range(1, n + 1):
            rows = math.ceil(n / cols)
            cw, ch = (W - 2 * m - gap * (cols - 1)) / cols, (H - 2 * m - gap * (rows - 1)) / rows
            score = min(cw / 1.4, ch)
            if best is None or score > best[0]:
                best = (score, cols, rows, cw, ch)
        _, cols, rows, cw, ch = best
        for i, c in enumerate(items):
            r, k = divmod(i, cols)
            c.setGeometry(int(m + k * (cw + gap)), int(m + r * (ch + gap)), int(cw), int(ch))


def cc_menu(widget, engine, key, pos):
    """Right-click menu: assign / clear a MIDI CC for a synth control."""
    m = QtWidgets.QMenu(widget)
    cc = [c for c, k in engine.cfg["cc_map"].items() if k == key]
    learn = m.addAction("Assign MIDI CC…" + (f"   (CC {cc[0]})" if cc else ""))
    clear = m.addAction("Clear CC") if cc else None
    act = m.exec(widget.mapToGlobal(pos))
    if act == learn:
        engine.learn = key
    elif clear is not None and act == clear:
        for c in cc:
            del engine.cfg["cc_map"][c]


def learn_blink(engine, key):
    return engine.learn == key and int(time.monotonic() * 3) % 2 == 0


class Knob(QtWidgets.QWidget):
    def __init__(self, key, label, default, engine):
        super().__init__()
        self.key, self.label, self.default, self.engine = key, label, default, engine
        self.setFixedSize(64, 76)
        self.drag = None
        self.hover = False
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: cc_menu(self, engine, key, pos))

    @property
    def value(self):
        return self.engine.knobs[self.key]

    def set(self, v):
        self.engine.knobs[self.key] = max(0.0, min(1.0, v))
        self.update()

    def enterEvent(self, e):
        self.hover = True

    def leaveEvent(self, e):
        self.hover = False

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.drag = (e.position().y(), self.value)

    def mouseMoveEvent(self, e):
        if self.drag:
            y0, v0 = self.drag
            scale = 600 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 150
            self.set(v0 + (y0 - e.position().y()) / scale)

    def mouseReleaseEvent(self, e):
        self.drag = None

    def mouseDoubleClickEvent(self, e):
        self.set(self.default)

    def wheelEvent(self, e):
        self.set(self.value + 0.02 * (1 if e.angleDelta().y() > 0 else -1))

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = QtCore.QRectF(12, 6, 40, 40)
        learning = self.engine.learn == self.key
        p.setPen(QtGui.QPen(FAINT, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(r, -45 * 16, 270 * 16)
        col = DIM if learn_blink(self.engine, self.key) else ACCENT
        p.setPen(QtGui.QPen(col, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(r, 225 * 16, int(-270 * 16 * self.value))
        p.setPen(TEXT)
        p.setFont(font(10))
        if self.drag or self.hover:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, knob_text(self.key, self.value))
        p.setPen(DIM)
        p.setFont(font(11))
        p.drawText(QtCore.QRectF(0, 52, 64, 20), Qt.AlignmentFlag.AlignCenter,
                   "learn…" if learning else self.label)
        p.end()


def wave_shape(name, x):
    ph = (1.5 * x) % 1.0
    if name == "saw":
        return 2 * ph - 1
    if name == "square":
        return 1.0 if ph < 0.5 else -1.0
    if name == "triangle":
        return 1 - 4 * abs(ph - 0.5)
    return math.sin(2 * math.pi * 1.5 * x)


class WaveSelector(QtWidgets.QWidget):
    """Four waveform icons; click one. Right-click to assign a MIDI CC."""
    SIZE, GAP = 30, 4

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.setFixedSize(4 * self.SIZE + 3 * self.GAP, 76)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: cc_menu(self, engine, "wave", pos))

    def index(self):
        return min(3, int(self.engine.knobs["wave"] * 4))

    def select(self, i):
        self.engine.knobs["wave"] = (max(0, min(3, i)) + 0.5) / 4
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.select(int(e.position().x() // (self.SIZE + self.GAP)))

    def wheelEvent(self, e):
        self.select(self.index() + (1 if e.angleDelta().y() < 0 else -1))

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        sel = self.index()
        for i, name in enumerate(WAVES):
            r = QtCore.QRectF(i * (self.SIZE + self.GAP), 11, self.SIZE, self.SIZE)
            if i == sel:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(FAINT)
                p.drawRoundedRect(r, 7, 7)
            path = QtGui.QPainterPath()
            for k in range(65):
                x = k / 64
                pt = QtCore.QPointF(r.left() + 6 + x * (r.width() - 12),
                                    r.center().y() - wave_shape(name, x) * (r.height() / 2 - 9))
                path.moveTo(pt) if k == 0 else path.lineTo(pt)
            col = ACCENT if i == sel else DIM
            if i == sel and learn_blink(self.engine, "wave"):
                col = DIM
            p.setPen(QtGui.QPen(col, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        p.setPen(DIM)
        p.setFont(font(11))
        p.drawText(QtCore.QRectF(0, 52, self.width(), 20), Qt.AlignmentFlag.AlignCenter,
                   "learn…" if self.engine.learn == "wave" else "Wave")
        p.end()


class Drawer(QtWidgets.QWidget):
    """Synth controls in a bottom drawer. Click or drag the handle to lift / lower it."""
    HANDLE, BODY = 22, 86

    def __init__(self, engine, is_open, on_toggle):
        super().__init__()
        self.is_open, self.on_toggle = is_open, on_toggle
        self.body = QtWidgets.QWidget(self)
        lay = QtWidgets.QHBoxLayout(self.body)
        lay.setContentsMargins(18, 0, 18, 6)
        lay.setSpacing(6)
        lay.addStretch()
        self.wave = WaveSelector(engine)
        lay.addWidget(self.wave)
        lay.addSpacing(18)
        self.knobs = [Knob(k, l, d, engine) for k, l, d in KNOBS]
        for k in self.knobs:
            lay.addWidget(k)
        lay.addStretch()
        self.setFixedHeight(self.target())
        self.anim = QtCore.QVariantAnimation(self)
        self.anim.setDuration(220)
        self.anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.anim.valueChanged.connect(lambda v: self.setFixedHeight(int(v)))
        self.press = None
        self.on_handle = False
        self.setMouseTracking(True)

    def target(self):
        return self.HANDLE + (self.BODY if self.is_open else 0)

    def set_open(self, is_open):
        self.is_open = is_open
        self.anim.stop()
        self.anim.setStartValue(self.height())
        self.anim.setEndValue(self.target())
        self.anim.start()
        self.on_toggle(is_open)

    def toggle(self):
        self.set_open(not self.is_open)

    def resizeEvent(self, e):
        self.body.setGeometry(0, self.HANDLE, self.width(), self.BODY)

    def mousePressEvent(self, e):
        if e.position().y() <= self.HANDLE:
            self.anim.stop()
            self.press = (e.globalPosition().y(), self.height(), False)

    def mouseMoveEvent(self, e):
        on = e.position().y() <= self.HANDLE
        if on != self.on_handle:
            self.on_handle = on
            self.setCursor(Qt.CursorShape.OpenHandCursor if on else Qt.CursorShape.ArrowCursor)
            self.update()
        if self.press:
            y0, h0, _ = self.press
            dy = y0 - e.globalPosition().y()
            self.press = (y0, h0, abs(dy) > 4)
            self.setFixedHeight(int(max(self.HANDLE, min(self.HANDLE + self.BODY, h0 + dy))))

    def mouseReleaseEvent(self, e):
        if self.press:
            dragged = self.press[2]
            self.press = None
            self.set_open(self.height() > self.HANDLE + self.BODY / 2 if dragged else not self.is_open)

    def leaveEvent(self, e):
        self.on_handle = False
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor("#111114"))
        p.drawRoundedRect(QtCore.QRectF(0, 0, self.width(), self.height() + 20), 16, 16)
        p.setBrush(TEXT if self.on_handle or self.press else QtGui.QColor("#3a3a42"))
        p.drawRoundedRect(QtCore.QRectF(self.width() / 2 - 18, 9, 36, 4), 2, 2)
        p.end()


class TopBar(QtWidgets.QWidget):
    settings = QtCore.Signal()

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setFixedHeight(46)
        self.gear = QtWidgets.QPushButton("⚙", self)
        self.gear.setFlat(True)
        self.gear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.gear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.gear.setStyleSheet("QPushButton{color:#6e6e78;font-size:18px;border:none;background:none}"
                                "QPushButton:hover{color:#f2f2f5}")
        self.gear.clicked.connect(self.settings)

    def resizeEvent(self, e):
        self.gear.setGeometry(self.width() - 46, 6, 34, 34)

    def paintEvent(self, e):
        app, cfg, w = self.app, self.app.cfg, self.width()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setFont(font(13))
        if app.engine.status or app.models_error:
            p.setPen(BAD)
            left = app.engine.status or app.models_error
        else:
            p.setPen(DIM)
            left = f"{'MIDI' if cfg['source'] == 'midi' else 'Audio'} · {cfg['mode']}"
        p.drawText(QtCore.QRectF(20, 0, w / 2 - 90, 46), Qt.AlignmentFlag.AlignVCenter, left)

        truth = app.engine.true_chord()
        if truth:
            root, qual = pretty(truth) if ":" in truth else (truth, "")
            p.setPen(TEXT)
            p.setFont(font(16, QtGui.QFont.Weight.Medium))
            p.drawText(QtCore.QRectF(0, 0, w, 46), Qt.AlignmentFlag.AlignCenter, f"{root} {qual}".strip())

        # level meter with gate tick
        mx, mw = w - 130, 70
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(FAINT)
        p.drawRoundedRect(QtCore.QRectF(mx, 21, mw, 4), 2, 2)
        frac = lambda db: max(0.0, min(1.0, (db + 70) / 70))
        lvl = app.inference.level_db
        p.setBrush(ACCENT if lvl > cfg["gate_db"] else DIM)
        p.drawRoundedRect(QtCore.QRectF(mx, 21, mw * frac(lvl), 4), 2, 2)
        p.setBrush(TEXT)
        p.drawRect(QtCore.QRectF(mx + mw * frac(cfg["gate_db"]), 18, 1, 10))

        prog = app.inference.progress
        if prog is not None:
            p.setBrush(ACCENT)
            p.drawRect(QtCore.QRectF(0, 44, w * prog, 2))
        p.end()


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input")
        self.cfg = cfg
        devs = sd.query_devices()
        self.inputs = [d["name"] for d in devs if d["max_input_channels"] > 0]
        self.outputs = [d["name"] for d in devs if d["max_output_channels"] > 0]
        midis = mido.get_input_names()

        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(24, 24, 24, 20)
        form.setSpacing(12)
        self.source = self.combo(["MIDI synth", "Audio input"], 0 if cfg["source"] == "midi" else 1)
        self.mode = self.combo(["Live", "Strum"], 0 if cfg["mode"] == "live" else 1)
        self.midi = self.combo(["None"] + midis, (midis.index(cfg["midi_in"]) + 1) if cfg["midi_in"] in midis
                               else (1 if midis else 0))
        self.out = self.combo(["Default"] + self.outputs,
                              self.outputs.index(cfg["audio_out"]) + 1 if cfg["audio_out"] in self.outputs else 0)
        self.inp = self.combo(["Default"] + self.inputs,
                              self.inputs.index(cfg["audio_in"]) + 1 if cfg["audio_in"] in self.inputs else 0)
        self.channel = QtWidgets.QComboBox()
        self.inp.currentIndexChanged.connect(self.fill_channels)
        self.fill_channels()
        self.channel.setCurrentIndex(min(cfg["channel"], self.channel.count() - 1))
        self.gate = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.gate.setRange(-80, -10)
        self.gate.setValue(cfg["gate_db"])
        self.gate_lbl = QtWidgets.QLabel()
        self.gate.valueChanged.connect(lambda v: self.gate_lbl.setText(f"{v} dB"))
        self.gate_lbl.setText(f"{cfg['gate_db']} dB")
        gate_row = QtWidgets.QHBoxLayout()
        gate_row.addWidget(self.gate)
        gate_row.addWidget(self.gate_lbl)

        form.addRow("Input", self.source)
        form.addRow("Mode", self.mode)
        form.addRow("MIDI controller", self.midi)
        form.addRow("Synth output", self.out)
        form.addRow("Audio input", self.inp)
        form.addRow("Channel", self.channel)
        form.addRow("Gate", gate_row)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                          | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)
        self.form = form
        self.source.currentIndexChanged.connect(self.sync_rows)
        self.sync_rows()
        self.setMinimumWidth(420)

    def sync_rows(self):
        midi = self.source.currentIndex() == 0
        self.form.setRowVisible(self.out, midi)
        self.form.setRowVisible(self.inp, not midi)
        self.form.setRowVisible(self.channel, not midi)
        self.adjustSize()

    @staticmethod
    def combo(items, idx):
        c = QtWidgets.QComboBox()
        c.addItems(items)
        c.setCurrentIndex(max(0, idx))
        return c

    def fill_channels(self):
        i = self.inp.currentIndex()
        try:
            dev = sd.query_devices(self.inputs[i - 1] if i > 0 else None, "input")
            n = dev["max_input_channels"]
        except Exception:  # noqa: BLE001
            n = 1
        self.channel.clear()
        self.channel.addItems(["Mix"] + [str(k) for k in range(1, n + 1)])

    def apply(self):
        c = self.cfg
        c["source"] = "midi" if self.source.currentIndex() == 0 else "audio"
        c["mode"] = "live" if self.mode.currentIndex() == 0 else "strum"
        c["midi_in"] = self.midi.currentText() if self.midi.currentIndex() > 0 else None
        c["audio_out"] = self.out.currentText() if self.out.currentIndex() > 0 else None
        c["audio_in"] = self.inp.currentText() if self.inp.currentIndex() > 0 else None
        c["channel"] = self.channel.currentIndex()
        c["gate_db"] = self.gate.value()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.setWindowTitle("Chords")
        self.resize(1200, 760)
        self.engine = Engine(self.cfg)
        self.slots = []
        self.inference = Inference(self.engine, self.slots)

        self.top = TopBar(self)
        self.top.settings.connect(self.open_settings)
        self.grid = Grid(self.model_names)
        self.grid.add.added.connect(self.add_model)
        self.drawer = Drawer(self.engine, self.cfg.get("synth_open", True), self.on_drawer)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.top)
        lay.addWidget(self.grid, 1)
        lay.addWidget(self.drawer)

        self.models_error = import_models()
        for name in self.cfg["models"]:
            self.add_model(name, save=False)
        self.watcher = QtCore.QFileSystemWatcher([str(MODELS_FILE)], self)
        self.watcher.fileChanged.connect(lambda _: QtCore.QTimer.singleShot(300, self.reload_models))
        self.engine.start()
        self.inference.start()
        self.sync_panels()

        for key, fn in [("Space", self.toggle_mode), ("S", self.open_settings), ("K", self.toggle_knobs),
                        ("F", self.toggle_fullscreen), ("R", self.reload_models), ("Escape", self.showNormal)]:
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)
        self.timer = QtCore.QTimer(self, timeout=self.refresh, interval=33)
        self.timer.start()

    # -- models --------------------------------------------------------------
    def model_names(self):
        reg = getattr(pm, "Model", None)
        return [m.name for m in reg.registry.values()] if reg else []

    def add_model(self, name, save=True):
        slot = Slot(name)
        self.slots.append(slot)                   # same list object the inference thread reads
        cell = ChordCell(slot, self)
        cell.removed.connect(self.remove_cell)
        cell.renamed.connect(lambda _: self.save_models())
        self.grid.set_cells(self.grid.cells + [cell])
        if save:
            self.save_models()

    def remove_cell(self, cell):
        self.slots.remove(cell.slot)
        self.grid.set_cells([c for c in self.grid.cells if c is not cell])
        self.save_models()

    def save_models(self):
        self.cfg["models"] = [s.name for s in self.slots]
        save_config(self.cfg)

    def reload_models(self):
        self.models_error = import_models()
        for s in self.slots:
            s.reset()
        if MODELS_FILE.exists() and str(MODELS_FILE) not in self.watcher.files():
            self.watcher.addPath(str(MODELS_FILE))   # editors that save by replacing the file

    # -- settings ------------------------------------------------------------
    def open_settings(self):
        d = SettingsDialog(self.cfg, self)
        if d.exec():
            d.apply()
            self.engine.start()
            self.sync_panels()
            save_config(self.cfg)

    def sync_panels(self):
        self.drawer.setVisible(self.cfg["source"] == "midi")

    def toggle_mode(self):
        self.cfg["mode"] = "strum" if self.cfg["mode"] == "live" else "live"
        self.engine.onset = None

    def toggle_knobs(self):
        self.drawer.toggle()

    def on_drawer(self, is_open):
        self.cfg["synth_open"] = is_open

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def refresh(self):
        self.top.update()
        for c in self.grid.cells:
            c.update()
        if self.drawer.isVisible() and self.drawer.height() > Drawer.HANDLE:
            self.drawer.wave.update()
            for k in self.drawer.knobs:
                k.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), BG)

    def closeEvent(self, e):
        self.inference.running = False
        self.engine.stop()
        save_config(self.cfg)
        super().closeEvent(e)


def dark_palette(app):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in [("Window", "#141417"), ("WindowText", "#e6e6ea"), ("Base", "#0f0f12"),
                      ("AlternateBase", "#18181c"), ("Text", "#e6e6ea"), ("Button", "#1c1c21"),
                      ("ButtonText", "#e6e6ea"), ("Highlight", "#8fa8ff"), ("HighlightedText", "#0b0b0d"),
                      ("ToolTipBase", "#1c1c21"), ("ToolTipText", "#e6e6ea")]:
        pal.setColor(getattr(QtGui.QPalette.ColorRole, role), QtGui.QColor(col))
    app.setPalette(pal)
    app.setStyleSheet("QMenu{background:#18181c;color:#e6e6ea;border:1px solid #26262c;padding:4px}"
                      "QMenu::item{padding:6px 18px;border-radius:4px}"
                      "QMenu::item:selected{background:#26262c}"
                      "QToolTip{background:#1c1c21;color:#e6e6ea;border:1px solid #26262c}")


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    dark_palette(app)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
