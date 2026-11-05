"""Procedural sound-effect synthesizer.

Every effect is generated from scratch with numpy (oscillators, filtered noise,
envelopes), so it works fully offline, has no licensing issues and can be
rendered at any duration. Each generator takes ``(duration, rng)`` and returns
mono float32 at ``SAMPLE_RATE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import zlib

import numpy as np

from ..audio import fade, normalize
from ..config import SAMPLE_RATE as SR

Generator = Callable[[float, np.random.Generator], np.ndarray]


# --------------------------------------------------------------------------- building blocks

def _t(duration: float) -> np.ndarray:
    return np.arange(int(duration * SR), dtype=np.float32) / SR


def _sine(freq, t: np.ndarray) -> np.ndarray:
    """Sine with constant or time-varying frequency (phase-accumulated)."""
    if np.isscalar(freq):
        return np.sin(2 * np.pi * freq * t).astype(np.float32)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase).astype(np.float32)


def _exp_env(t: np.ndarray, decay: float) -> np.ndarray:
    return np.exp(-t / max(decay, 1e-4)).astype(np.float32)


def _adsr(n: int, a: float = 0.01, d: float = 0.05, s: float = 0.7, r: float = 0.1) -> np.ndarray:
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    s_n = max(0, n - a_n - d_n - r_n)
    env = np.concatenate([
        np.linspace(0, 1, a_n, endpoint=False), np.linspace(1, s, d_n, endpoint=False),
        np.full(s_n, s), np.linspace(s, 0, r_n),
    ]).astype(np.float32)
    return np.pad(env, (0, max(0, n - len(env))))[:n]


def _fast_lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    """FFT brick-wall low-pass - fast for long noise beds."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec[freqs > cutoff] = 0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spec, n=len(x)).astype(np.float32)


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1, 1, n).astype(np.float32)


def _brown(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.cumsum(rng.normal(0, 1, n)).astype(np.float32)
    x -= np.convolve(x, np.ones(2048) / 2048, mode="same").astype(np.float32)  # remove DC drift
    return normalize(x, 1.0)


def _place(buf: np.ndarray, sound: np.ndarray, at: float, gain: float = 1.0) -> None:
    i = int(at * SR)
    if i >= len(buf):
        return
    end = min(len(buf), i + len(sound))
    buf[i:end] += sound[: end - i] * gain


def _bell_partials(t: np.ndarray, f0: float, ratios, decays, amps) -> np.ndarray:
    out = np.zeros_like(t)
    for r, d, a in zip(ratios, decays, amps):
        out += a * _sine(f0 * r, t) * _exp_env(t, d)
    return out


# --------------------------------------------------------------------------- UI / notification sounds

def beep(d: float, rng) -> np.ndarray:
    t = _t(d)
    return _sine(1000, t) * _adsr(len(t), 0.005, 0.02, 0.8, 0.03)


def chime(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    for i, f in enumerate((1046.5, 1318.5, 1568.0)):  # C6 E6 G6 arpeggio
        tt = _t(max(0.01, d - i * 0.12))
        _place(out, _bell_partials(tt, f, (1, 2.0, 3.01), (0.6, 0.3, 0.15), (1, 0.4, 0.2)), i * 0.12)
    return out


def notification(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    for i, f in enumerate((880, 1320)):
        tt = _t(0.35)
        _place(out, _sine(f, tt) * _exp_env(tt, 0.12), i * 0.1)
    return out


def success(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    for i, f in enumerate((523.25, 659.25, 783.99, 1046.5)):
        tt = _t(0.4)
        _place(out, (_sine(f, tt) + 0.3 * _sine(2 * f, tt)) * _exp_env(tt, 0.18), i * 0.09)
    return out


def error(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    for i in range(2):
        tt = _t(0.18)
        tone = np.sign(_sine(180, tt)) * 0.5 + _sine(185, tt) * 0.5
        _place(out, tone * _adsr(len(tt), 0.005, 0.02, 0.8, 0.04), i * 0.22)
    return out


def click(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    _place(out, _noise(int(0.004 * SR), rng) * np.linspace(1, 0, int(0.004 * SR)), 0)
    return out


def pop(d: float, rng) -> np.ndarray:
    t = _t(min(d, 0.25))
    f = 900 * np.exp(-t / 0.02) + 200
    return np.pad(_sine(f, t) * _exp_env(t, 0.04), (0, max(0, int(d * SR) - len(t))))


def typing(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    at = 0.0
    key = int(0.012 * SR)
    while at < d - 0.05:
        k = _bandpass(_noise(key * 4, rng), 1500, 6000)[:key] * np.linspace(1, 0, key)
        _place(out, k, at, rng.uniform(0.5, 1.0))
        at += rng.uniform(0.06, 0.18)
    return out


# --------------------------------------------------------------------------- transitions

def whoosh(d: float, rng) -> np.ndarray:
    t = _t(d)
    n = _noise(len(t), rng)
    # Sweep a band upward by blending filtered layers with moving weights.
    low, high = _bandpass(n, 200, 1200), _bandpass(n, 1200, 6000)
    p = t / max(d, 1e-3)
    env = np.sin(np.pi * p) ** 2
    return (low * (1 - p) + high * p) * env


def swoosh_down(d: float, rng) -> np.ndarray:
    return whoosh(d, rng)[::-1].copy()


def riser(d: float, rng) -> np.ndarray:
    t = _t(d)
    f = 200 * (2 ** (3 * t / d))
    p = t / d
    return (_sine(f, t) * 0.5 + _bandpass(_noise(len(t), rng), 2000, 8000) * 0.3 * p) * p ** 2


def magic(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    scale = [1046.5, 1174.7, 1318.5, 1568.0, 1760.0, 2093.0, 2349.3, 2637.0]
    for _ in range(int(d * 22)):
        f = scale[rng.integers(len(scale))] * rng.choice([1, 2])
        tt = _t(0.25)
        _place(out, _sine(f, tt) * _exp_env(tt, 0.07), rng.uniform(0, max(0.01, d - 0.25)), rng.uniform(0.2, 0.6))
    return out * np.linspace(1, 0.3, len(out)).astype(np.float32)


def laser(d: float, rng) -> np.ndarray:
    d = min(d, 0.6)
    t = _t(d)
    f = 2400 * np.exp(-t / 0.08) + 180
    return np.sign(_sine(f, t)) * 0.4 * _exp_env(t, 0.15) + _sine(f, t) * 0.4 * _exp_env(t, 0.15)


# --------------------------------------------------------------------------- nature / ambience

def rain(d: float, rng) -> np.ndarray:
    t = _t(d)
    bed = _fast_lowpass(_noise(len(t), rng), 5000) * 0.35
    drops = np.zeros_like(t)
    drop_len = int(0.006 * SR)
    for at in rng.uniform(0, d, int(d * 120)):
        drop = _bandpass(_noise(drop_len * 4, rng), 2000, 9000)[:drop_len] * np.linspace(1, 0, drop_len)
        _place(drops, drop, at, rng.uniform(0.1, 0.6))
    return bed + drops


def thunder(d: float, rng) -> np.ndarray:
    t = _t(d)
    crack = _bandpass(_noise(len(t), rng), 800, 6000) * _exp_env(t, 0.08)
    rumble = _fast_lowpass(_brown(len(t), rng), 180)
    rumble_env = (1 - np.exp(-t / 0.05)) * _exp_env(t, d / 3)
    # Slow amplitude wobble = distant rolling rumble.
    wobble = 0.6 + 0.4 * _fast_lowpass(_noise(len(t), rng), 4) / 0.05
    return crack * 0.6 + normalize(rumble, 1.0) * rumble_env * np.clip(wobble, 0.2, 1.5)


def wind(d: float, rng) -> np.ndarray:
    t = _t(d)
    gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t + rng.uniform(0, 6)) * np.sin(2 * np.pi * 0.37 * t)
    return _bandpass(_noise(len(t), rng), 150, 900) * (0.3 + 0.7 * gust)


def ocean(d: float, rng) -> np.ndarray:
    t = _t(d)
    period = 5.5
    swell = (0.5 - 0.5 * np.cos(2 * np.pi * t / period)) ** 2
    surf = _fast_lowpass(_noise(len(t), rng), 2500) * swell
    undertow = _fast_lowpass(_brown(len(t), rng), 300) * 0.4
    return surf + undertow


def fire(d: float, rng) -> np.ndarray:
    t = _t(d)
    roar = _fast_lowpass(_brown(len(t), rng), 400) * 0.5
    crackles = np.zeros_like(t)
    for at in rng.uniform(0, d, int(d * 25)):
        c = _noise(int(0.003 * SR), rng) * np.linspace(1, 0, int(0.003 * SR))
        _place(crackles, c, at, rng.uniform(0.3, 1.0))
    return roar + crackles


def birds(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    at = 0.1
    while at < d - 0.4:
        notes = rng.integers(2, 6)
        base = rng.uniform(2500, 4500)
        for k in range(notes):
            tt = _t(rng.uniform(0.05, 0.12))
            f = base + 1200 * np.sin(2 * np.pi * rng.uniform(8, 20) * tt) + rng.uniform(-300, 300)
            _place(out, _sine(f, tt) * np.sin(np.pi * tt / tt[-1]) ** 2, at + k * 0.13, 0.5)
        at += rng.uniform(0.6, 1.6)
    return out


def crickets(d: float, rng) -> np.ndarray:
    t = _t(d)
    chirp_gate = (np.sin(2 * np.pi * 30 * t) > 0.2) * (np.sin(2 * np.pi * 1.2 * t) > 0)
    return _sine(4400, t) * chirp_gate.astype(np.float32) * 0.4


# --------------------------------------------------------------------------- people / actions

def applause(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    clap_len = int(0.012 * SR)
    for at in rng.uniform(0, d, int(d * 90)):
        c = _bandpass(_noise(clap_len * 4, rng), 700, 5000)[:clap_len] * _exp_env(_t(clap_len / SR), 0.004)
        _place(out, c, at, rng.uniform(0.3, 1.0))
    env = np.minimum(1, t / 0.3) * np.minimum(1, (d - t) / 0.8)
    return out * env


def heartbeat(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    beat = 60 / 72
    at = 0.05
    while at < d:
        for off, g in ((0, 1.0), (0.18, 0.7)):
            tt = _t(0.12)
            _place(out, _sine(55 * np.exp(-tt / 0.05) + 40, tt) * _exp_env(tt, 0.04), at + off, g)
        at += beat
    return out


def footsteps(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    at = 0.05
    while at < d - 0.1:
        tt = _t(0.09)
        step = _fast_lowpass(_noise(len(tt), rng), 1200) * _exp_env(tt, 0.025) * 3
        step += _sine(90, tt) * _exp_env(tt, 0.03)
        _place(out, step, at, rng.uniform(0.7, 1.0))
        at += rng.uniform(0.48, 0.56)
    return out


def knock(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    for i, at in enumerate((0.05, 0.25, 0.45)):
        tt = _t(0.1)
        k = _sine(220 * np.exp(-tt / 0.02) + 150, tt) * _exp_env(tt, 0.03)
        k += _bandpass(_noise(len(tt), rng), 400, 2500) * _exp_env(tt, 0.01) * 0.6
        _place(out, k, at)
    return out


def door_creak(d: float, rng) -> np.ndarray:
    t = _t(d)
    f = 280 + 120 * np.sin(2 * np.pi * 0.6 * t) + 40 * _fast_lowpass(_noise(len(t), rng), 10) / 0.05
    saw = 2 * ((np.cumsum(f) / SR) % 1) - 1
    return _bandpass(saw.astype(np.float32), 300, 3000) * np.sin(np.pi * t / d) ** 0.5


def clock_tick(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    at, i = 0.05, 0
    while at < d:
        tt = _t(0.03)
        _place(out, _sine(3200 if i % 2 else 2600, tt) * _exp_env(tt, 0.005), at)
        at += 0.5
        i += 1
    return out


def phone_ring(d: float, rng) -> np.ndarray:
    t = _t(d)
    ring = (_sine(440, t) + _sine(480, t)) * 0.5
    gate = ((t % 3.0) < 1.0) * (np.sign(_sine(20, t)) * 0.5 + 0.5)
    return ring * gate.astype(np.float32)


# --------------------------------------------------------------------------- music / cultural

def bell(d: float, rng) -> np.ndarray:
    t = _t(d)
    return _bell_partials(t, 523.25, (0.5, 1, 1.19, 1.56, 2.0, 2.74, 3.0), (2.5, 2.0, 1.5, 1.0, 0.8, 0.5, 0.4),
                          (0.6, 1, 0.5, 0.4, 0.3, 0.2, 0.15))


def temple_bell(d: float, rng) -> np.ndarray:
    """Brass temple bell (గుడి గంట): a few strikes with long inharmonic decay."""
    t = _t(d)
    out = np.zeros_like(t)
    strikes = max(1, min(4, int(d / 0.9)))
    for i in range(strikes):
        tt = _t(max(0.1, d - i * 0.8))
        tone = _bell_partials(tt, 740, (1, 2.32, 4.25, 5.4, 6.8), (2.2, 1.4, 0.8, 0.5, 0.3), (1, 0.6, 0.4, 0.3, 0.2))
        tone *= 1 + 0.08 * np.sin(2 * np.pi * 5 * tt)  # shimmer
        _place(out, tone, i * 0.8)
    return out


def conch(d: float, rng) -> np.ndarray:
    """Conch shell (శంఖం): breathy harmonic drone with a pitch bend."""
    t = _t(d)
    f = 233 * (1 + 0.03 * np.minimum(1, t / 0.6)) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
    tone = sum(a * _sine(f * k, t) for k, a in ((1, 1.0), (2, 0.6), (3, 0.35), (4, 0.2), (5, 0.1)))
    breath = _bandpass(_noise(len(t), rng), 800, 4000) * 0.15
    return (tone + breath) * _adsr(len(t), 0.4, 0.2, 0.85, min(0.8, d / 3))


def drum(d: float, rng) -> np.ndarray:
    """Folk frame-drum (డప్పు) groove."""
    t = _t(d)
    out = np.zeros_like(t)
    pattern = [(0, 1.0, "low"), (0.25, 0.5, "slap"), (0.375, 0.6, "slap"), (0.5, 0.9, "low"), (0.75, 0.6, "slap")]
    bar = 1.0
    for b in range(int(np.ceil(d / bar))):
        for off, g, kind in pattern:
            tt = _t(0.25)
            if kind == "low":
                hit = _sine(110 * np.exp(-tt / 0.04) + 70, tt) * _exp_env(tt, 0.12)
            else:
                hit = _bandpass(_noise(len(tt), rng), 400, 4000) * _exp_env(tt, 0.03) + _sine(330, tt) * _exp_env(tt, 0.04) * 0.4
            _place(out, hit, b * bar + off, g)
    return out


def drumroll(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    at = 0.0
    hit_len = int(0.05 * SR)
    while at < d - 0.3:
        h = _bandpass(_noise(hit_len, rng), 150, 5000) * _exp_env(_t(hit_len / SR), 0.015)
        _place(out, h, at, 0.4 + 0.6 * at / d)
        at += 0.035
    tt = _t(0.3)
    _place(out, _sine(90, tt) * _exp_env(tt, 0.1) + _noise(len(tt), rng) * _exp_env(tt, 0.08) * 0.5, d - 0.3, 1.2)
    return out


def fanfare(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = np.zeros_like(t)
    notes = [(0, 392, 0.15), (0.17, 392, 0.15), (0.34, 392, 0.15), (0.51, 523.25, 0.9)]
    for at, f, length in notes:
        tt = _t(min(length, max(0.05, d - at)))
        brass = sum(a * _sine(f * k, tt) for k, a in ((1, 1), (2, 0.7), (3, 0.5), (4, 0.3), (5, 0.2)))
        _place(out, brass * _adsr(len(tt), 0.02, 0.05, 0.8, 0.1), at)
    return out


def sitar_strum(d: float, rng) -> np.ndarray:
    """Plucked string (Karplus-Strong) arpeggio with sympathetic buzz."""
    out = np.zeros(int(d * SR), dtype=np.float32)
    for i, f in enumerate((146.8, 220.0, 293.7, 440.0)):
        n = max(1, int(SR / f))
        buf = rng.uniform(-1, 1, n).astype(np.float32)
        length = int(min(d, 2.5) * SR)
        y = np.empty(length, dtype=np.float32)
        for k in range(length):
            y[k] = buf[k % n]
            buf[k % n] = 0.996 * 0.5 * (buf[k % n] + buf[(k + 1) % n])
        _place(out, np.tanh(y * 2.5) * 0.5, i * 0.12)
    return out


# --------------------------------------------------------------------------- impacts

def explosion(d: float, rng) -> np.ndarray:
    t = _t(d)
    body = _fast_lowpass(_brown(len(t), rng), 250) * _exp_env(t, d / 3)
    blast = _fast_lowpass(_noise(len(t), rng), 3000) * _exp_env(t, 0.12)
    thump = _sine(50 * np.exp(-t / 0.2) + 30, t) * _exp_env(t, 0.3)
    return normalize(body, 1) * 0.8 + blast * 0.7 + thump


def punch(d: float, rng) -> np.ndarray:
    t = _t(min(d, 0.4))
    hit = _sine(120 * np.exp(-t / 0.03) + 60, t) * _exp_env(t, 0.06) + _noise(len(t), rng) * _exp_env(t, 0.015) * 0.6
    return np.pad(hit, (0, max(0, int(d * SR) - len(t))))


def glass_break(d: float, rng) -> np.ndarray:
    t = _t(d)
    out = _bandpass(_noise(len(t), rng), 3000, 11000) * _exp_env(t, 0.08)
    for _ in range(14):
        tt = _t(0.4)
        _place(out, _sine(rng.uniform(3000, 9000), tt) * _exp_env(tt, rng.uniform(0.05, 0.2)), rng.uniform(0, min(0.6, d)), 0.3)
    return out


# --------------------------------------------------------------------------- registry

@dataclass(frozen=True)
class EffectSpec:
    name: str
    generator: Generator
    default_duration: float
    category: str
    label_te: str
    keywords: tuple[str, ...]  # English + Telugu search terms
    loopable: bool = False


EFFECTS: dict[str, EffectSpec] = {e.name: e for e in [
    EffectSpec("beep", beep, 0.3, "ui", "బీప్", ("beep", "tone", "censor", "బీప్")),
    EffectSpec("chime", chime, 1.6, "ui", "చైమ్", ("chime", "ding", "welcome", "start", "మధుర ధ్వని")),
    EffectSpec("notification", notification, 0.7, "ui", "నోటిఫికేషన్", ("notification", "notify", "alert", "message", "సందేశం", "నోటిఫికేషన్")),
    EffectSpec("success", success, 0.9, "ui", "విజయం", ("success", "done", "complete", "correct", "win", "విజయం", "సరైనది")),
    EffectSpec("error", error, 0.5, "ui", "తప్పు", ("error", "wrong", "fail", "buzzer", "తప్పు", "లోపం")),
    EffectSpec("click", click, 0.1, "ui", "క్లిక్", ("click", "tap", "క్లిక్")),
    EffectSpec("pop", pop, 0.25, "ui", "పాప్", ("pop", "bubble", "బుడగ")),
    EffectSpec("typing", typing, 2.5, "ui", "టైపింగ్", ("typing", "keyboard", "type", "టైపింగ్", "కీబోర్డ్")),
    EffectSpec("whoosh", whoosh, 0.8, "transition", "వూష్", ("whoosh", "swish", "swipe", "transition", "fast", "వేగం")),
    EffectSpec("swoosh_down", swoosh_down, 0.8, "transition", "స్వూష్", ("swoosh", "reverse", "fall", "పడిపోవడం")),
    EffectSpec("riser", riser, 2.5, "transition", "రైజర్", ("riser", "rise", "build up", "tension", "suspense", "ఉత్కంఠ")),
    EffectSpec("magic", magic, 1.8, "transition", "మాయ", ("magic", "sparkle", "fairy", "twinkle", "మాయ", "మెరుపు", "అద్భుతం")),
    EffectSpec("laser", laser, 0.5, "transition", "లేజర్", ("laser", "zap", "sci-fi", "shoot", "లేజర్")),
    EffectSpec("rain", rain, 6.0, "nature", "వర్షం", ("rain", "raining", "drizzle", "storm", "వర్షం", "వాన", "జల్లు"), True),
    EffectSpec("thunder", thunder, 4.0, "nature", "ఉరుము", ("thunder", "lightning", "thunderstorm", "ఉరుము", "పిడుగు", "మెరుపులు"),),
    EffectSpec("wind", wind, 6.0, "nature", "గాలి", ("wind", "breeze", "windy", "gust", "గాలి", "పవనం"), True),
    EffectSpec("ocean", ocean, 11.0, "nature", "సముద్రం", ("ocean", "sea", "waves", "beach", "సముద్రం", "అలలు", "తీరం"), True),
    EffectSpec("fire", fire, 6.0, "nature", "నిప్పు", ("fire", "campfire", "crackle", "flame", "నిప్పు", "మంట", "అగ్ని"), True),
    EffectSpec("birds", birds, 6.0, "nature", "పక్షులు", ("birds", "bird", "chirp", "morning", "forest", "పక్షులు", "పిట్టలు", "ఉదయం", "అడవి"), True),
    EffectSpec("crickets", crickets, 6.0, "nature", "కీచురాళ్లు", ("crickets", "night", "insects", "కీచురాళ్లు", "రాత్రి"), True),
    EffectSpec("applause", applause, 4.0, "people", "చప్పట్లు", ("applause", "clap", "clapping", "audience", "cheer", "చప్పట్లు", "అభినందన")),
    EffectSpec("heartbeat", heartbeat, 4.0, "people", "గుండె చప్పుడు", ("heartbeat", "heart", "pulse", "nervous", "గుండె", "హృదయం"), True),
    EffectSpec("footsteps", footsteps, 4.0, "people", "అడుగులు", ("footsteps", "walking", "steps", "walk", "అడుగులు", "నడక"), True),
    EffectSpec("knock", knock, 0.8, "people", "తలుపు తట్టడం", ("knock", "door knock", "knocking", "తలుపు", "తట్టు")),
    EffectSpec("door_creak", door_creak, 2.0, "people", "తలుపు కిర్రు", ("creak", "door open", "creaky", "haunted", "spooky", "కిర్రు", "దెయ్యం")),
    EffectSpec("clock_tick", clock_tick, 4.0, "people", "గడియారం", ("clock", "tick", "ticking", "time", "timer", "గడియారం", "సమయం"), True),
    EffectSpec("phone_ring", phone_ring, 4.0, "people", "ఫోన్ రింగ్", ("phone", "ring", "ringing", "call", "ఫోన్", "కాల్")),
    EffectSpec("bell", bell, 3.0, "music", "గంట", ("bell", "church bell", "school bell", "గంట")),
    EffectSpec("temple_bell", temple_bell, 4.0, "music", "గుడి గంట", ("temple bell", "temple", "pooja", "puja", "prayer", "devotional", "గుడి", "ఆలయం", "పూజ", "గుడి గంట")),
    EffectSpec("conch", conch, 3.0, "music", "శంఖం", ("conch", "shankh", "shell horn", "శంఖం", "శంఖనాదం")),
    EffectSpec("drum", drum, 4.0, "music", "డప్పు", ("drum", "dappu", "beat", "folk", "festival", "celebration", "డప్పు", "డోలు", "పండుగ", "సంబరం"), True),
    EffectSpec("drumroll", drumroll, 2.5, "music", "డ్రమ్ రోల్", ("drumroll", "drum roll", "reveal", "announcement", "ప్రకటన")),
    EffectSpec("fanfare", fanfare, 1.8, "music", "విజయ నాదం", ("fanfare", "trumpet", "victory", "royal", "కొమ్ము", "రాజ")),
    EffectSpec("sitar", sitar_strum, 2.5, "music", "సితార్", ("sitar", "veena", "string", "pluck", "strum", "సితార్", "వీణ", "తీగ")),
    EffectSpec("explosion", explosion, 3.0, "impact", "పేలుడు", ("explosion", "explode", "boom", "blast", "bomb", "పేలుడు", "బాంబు", "ఢాం")),
    EffectSpec("punch", punch, 0.4, "impact", "గుద్దు", ("punch", "hit", "fight", "kick", "గుద్దు", "దెబ్బ", "కొట్టు")),
    EffectSpec("glass_break", glass_break, 1.5, "impact", "గాజు పగలడం", ("glass", "shatter", "break", "crash", "గాజు", "పగలడం")),
]}


def render(name: str, duration: float | None = None, seed: int | None = None) -> np.ndarray:
    """Render effect ``name`` (normalized, faded) for ``duration`` seconds."""
    spec = EFFECTS[name]
    dur = float(np.clip(duration if duration else spec.default_duration, 0.05, 30.0))
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(name.encode()))
    audio = np.asarray(spec.generator(dur, rng), dtype=np.float32)
    audio = np.nan_to_num(audio)
    audio = audio[: int(dur * SR)]
    if len(audio) < int(dur * SR):
        audio = np.pad(audio, (0, int(dur * SR) - len(audio)))
    return fade(normalize(audio, 0.9), 0.003, min(0.05, dur / 4))
