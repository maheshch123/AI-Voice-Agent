"""The production engine: script -> timeline -> mixed audio.

Speech segments are synthesized concurrently, then laid out on a timeline with
inline effects and pauses. Background beds are looped under the speech and
automatically ducked while someone is talking.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from . import audio as A
from .config import SAMPLE_RATE, get_settings
from .language import segment_text
from .script import Background, Effect, Pause, Speech, parse_script, strip_tags
from .sfx import SoundEffectLibrary
from .tts import get_provider, pick_voice
from .tts.base import Prosody

GAP_BETWEEN_SEGMENTS = 0.06  # s of air between language switches
BG_GAIN = 0.28


@dataclass
class VoiceOptions:
    gender: str = "female"  # female | male
    language: str = "auto"  # auto | te | en
    english_accent: str = "IN"  # IN | US | GB
    rate: int = 0
    pitch: int = 0
    volume: int = 0
    provider: str | None = None  # edge | elevenlabs (default from settings)
    sfx_provider: str = "auto"  # auto | procedural | elevenlabs


@dataclass
class TimelineItem:
    kind: str  # speech | sfx | pause | bg
    start: float
    duration: float
    detail: dict = field(default_factory=dict)


@dataclass
class Production:
    audio: np.ndarray
    timeline: list[TimelineItem]
    transcript: str

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class _LRU(OrderedDict):
    def __init__(self, size: int = 256):
        super().__init__()
        self.size = size

    def get_or_none(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)


class VoiceEngine:
    def __init__(self, sfx: SoundEffectLibrary | None = None, max_concurrency: int = 4):
        self.sfx = sfx or SoundEffectLibrary()
        self._tts_cache = _LRU()
        self._sem = asyncio.Semaphore(max_concurrency)

    # ------------------------------------------------------------------ speech
    async def speak(self, text: str, opts: VoiceOptions | None = None) -> tuple[np.ndarray, list[dict]]:
        """Synthesize plain (possibly code-mixed) text. Returns (audio, segments)."""
        opts = opts or VoiceOptions()
        if opts.language in ("te", "en"):
            from .language import Segment
            segments = [Segment(text=text.strip(), language=opts.language)]
        else:
            segments = segment_text(text)
        if not segments:
            return np.zeros(0, dtype=np.float32), []
        clips = await asyncio.gather(*(self._synth(s.text, s.language, opts.gender, opts) for s in segments))
        gap = A.silence(GAP_BETWEEN_SEGMENTS)
        parts: list[np.ndarray] = []
        info = []
        for seg, (clip, voice_id) in zip(segments, clips):
            if parts:
                parts.append(gap)
            parts.append(clip)
            info.append({"text": seg.text, "language": seg.language, "voice": voice_id})
        return np.concatenate(parts).astype(np.float32), info

    async def _synth(self, text: str, language: str, gender: str, opts: VoiceOptions) -> tuple[np.ndarray, str]:
        provider = get_provider(opts.provider or get_settings().tts_provider)
        voice = pick_voice(language, gender, opts.english_accent)
        prosody = Prosody(rate=opts.rate, pitch=opts.pitch, volume=opts.volume)
        key = (provider.name, text, voice.id, prosody)
        cached = self._tts_cache.get_or_none(key)
        if cached is not None:
            return cached, voice.id
        async with self._sem:
            clip = await provider.synthesize(text, voice, prosody)
        clip = A.fade(clip, 0.005, 0.02)
        self._tts_cache.put(key, clip)
        return clip, voice.id

    # ------------------------------------------------------------------ full production
    async def produce(self, script: str, opts: VoiceOptions | None = None) -> Production:
        opts = opts or VoiceOptions()
        events = parse_script(script)

        # Render every speech + sfx event concurrently; layout is sequential afterwards.
        async def render(ev):
            if isinstance(ev, Speech):
                o = VoiceOptions(**{**opts.__dict__,
                                    "gender": ev.voice or opts.gender,
                                    "language": ev.language or opts.language})
                return await self.speak(ev.text, o)
            if isinstance(ev, Effect):
                return await self.sfx.generate(ev.prompt, ev.duration, opts.sfx_provider)
            if isinstance(ev, Background) and ev.prompt:
                return await self.sfx.generate(ev.prompt, None, opts.sfx_provider)
            return None

        rendered = await asyncio.gather(*(render(ev) for ev in events))

        track = np.zeros(0, dtype=np.float32)
        voice_mask = np.zeros(0, dtype=np.float32)
        timeline: list[TimelineItem] = []
        beds: list[tuple[int, np.ndarray, dict]] = []  # (start, bed audio, info) - end set later
        bed_bounds: list[list[int]] = []
        cursor = 0

        def append(clip: np.ndarray, is_voice: bool):
            nonlocal track, voice_mask, cursor
            track = A.mix_at(track, clip, cursor)
            mask = np.full(len(clip), 1.0 if is_voice else 0.0, dtype=np.float32)
            voice_mask = A.mix_at(voice_mask, mask, cursor)
            cursor += len(clip)

        def close_bed():
            if bed_bounds and bed_bounds[-1][1] < 0:
                bed_bounds[-1][1] = cursor

        for ev, out in zip(events, rendered):
            start = cursor / SAMPLE_RATE
            if isinstance(ev, Speech):
                clip, segs = out
                append(clip, True)
                timeline.append(TimelineItem("speech", start, len(clip) / SAMPLE_RATE, {"text": ev.text, "segments": segs}))
                append(A.silence(0.12), False)
            elif isinstance(ev, Effect):
                clip, info = out
                append(clip, False)
                timeline.append(TimelineItem("sfx", start, len(clip) / SAMPLE_RATE, info))
                append(A.silence(0.08), False)
            elif isinstance(ev, Pause):
                append(A.silence(ev.seconds), False)
                timeline.append(TimelineItem("pause", start, ev.seconds))
            elif isinstance(ev, Background):
                close_bed()
                if ev.prompt:
                    clip, info = out
                    beds.append((cursor, clip, info))
                    bed_bounds.append([cursor, -1])
        close_bed()

        # Let a running bed ring out a little after the last word.
        if bed_bounds and bed_bounds[-1][1] == cursor:
            tail = A.silence(1.2)
            track = np.concatenate([track, tail])
            voice_mask = np.concatenate([voice_mask, np.zeros(len(tail), dtype=np.float32)])
            bed_bounds[-1][1] = len(track)

        for (start, clip, info), (_, end) in zip(beds, bed_bounds):
            n = end - start
            if n <= 0:
                continue
            bed = A.fade(A.fit_length(clip, n), 0.4, min(1.0, n / SAMPLE_RATE / 3))
            bed = A.duck(bed, voice_mask[start:end])
            track = A.mix_at(track, bed, start, BG_GAIN)
            timeline.append(TimelineItem("bg", start / SAMPLE_RATE, n / SAMPLE_RATE, info))

        if len(track):
            track = A.soft_limit(track)
        timeline.sort(key=lambda it: it.start)
        return Production(audio=track, timeline=timeline, transcript=strip_tags(script))
