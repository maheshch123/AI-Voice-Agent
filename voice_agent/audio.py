"""Small audio toolkit: float32 mono buffers at ``SAMPLE_RATE``.

Decoding of compressed audio (MP3 from TTS providers) and MP3 encoding use the
``ffmpeg`` binary. WAV encoding is pure Python so the engine works without it.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave

import numpy as np

from .config import SAMPLE_RATE


class AudioError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def silence(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sr))), dtype=np.float32)


def decode(data: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable audio (mp3, wav, ogg...) into mono float32."""
    if data[:4] == b"RIFF":
        try:
            return _decode_wav(data, sr)
        except (wave.Error, AudioError):
            pass
    if not ffmpeg_available():
        raise AudioError("ffmpeg is required to decode compressed audio. Install it (e.g. apt install ffmpeg).")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=data, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _decode_wav(data: bytes, sr: int) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise AudioError("only 16-bit PCM wav decoded natively")
        channels, rate = w.getnchannels(), w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return resample(pcm, rate, sr)


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * dst_sr / src_sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def fade(x: np.ndarray, fade_in: float = 0.01, fade_out: float = 0.02, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply short linear fades to avoid clicks at buffer edges."""
    x = x.astype(np.float32, copy=True)
    n_in = min(int(fade_in * sr), len(x))
    n_out = min(int(fade_out * sr), len(x))
    if n_in > 0:
        x[:n_in] *= np.linspace(0, 1, n_in, dtype=np.float32)
    if n_out > 0:
        x[-n_out:] *= np.linspace(1, 0, n_out, dtype=np.float32)
    return x


def normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m < 1e-9:
        return x.astype(np.float32)
    return (x * (peak / m)).astype(np.float32)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Loop or trim ``x`` to exactly ``n`` samples (used for background beds)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n].astype(np.float32)


def mix_at(base: np.ndarray, layer: np.ndarray, offset: int, gain: float = 1.0) -> np.ndarray:
    end = offset + len(layer)
    if end > len(base):
        base = np.concatenate([base, np.zeros(end - len(base), dtype=np.float32)])
    base[offset:end] += layer * gain
    return base


def duck(bed: np.ndarray, voice_mask: np.ndarray, amount: float = 0.45, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Lower ``bed`` while speech is active (sidechain-style ducking, smoothed)."""
    win = max(1, int(0.08 * sr))
    env = np.convolve(voice_mask.astype(np.float32), np.ones(win, dtype=np.float32) / win, mode="same")
    return (bed * (1.0 - amount * np.clip(env, 0, 1))).astype(np.float32)


def soft_limit(x: np.ndarray) -> np.ndarray:
    """Gentle tanh limiter so stacked layers never hard-clip."""
    return (np.tanh(x * 1.1) / np.tanh(1.1)).astype(np.float32)


def to_wav(x: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def to_mp3(x: np.ndarray, sr: int = SAMPLE_RATE, bitrate: str = "128k") -> bytes:
    if not ffmpeg_available():
        raise AudioError("ffmpeg is required for mp3 output; request wav instead.")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ac", "1", "-ar", str(sr),
         "-i", "pipe:0", "-b:a", bitrate, "-f", "mp3", "pipe:1"],
        input=np.clip(x, -1, 1).astype(np.float32).tobytes(), capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg encode failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def encode(x: np.ndarray, fmt: str = "wav") -> tuple[bytes, str]:
    """Encode to ``fmt`` and return (bytes, mime type)."""
    if fmt == "mp3":
        return to_mp3(x), "audio/mpeg"
    return to_wav(x), "audio/wav"
