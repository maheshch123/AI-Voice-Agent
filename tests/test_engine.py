import asyncio

import numpy as np

from voice_agent.config import SAMPLE_RATE
from voice_agent.engine import VoiceEngine, VoiceOptions


def test_speak_routes_languages_to_native_voices(fake_tts):
    audio, segs = asyncio.run(VoiceEngine().speak("నమస్కారం! Welcome to Vani."))
    assert [s["voice"] for s in segs] == ["te-IN-ShrutiNeural", "en-IN-NeerjaNeural"]
    assert len(audio) > 0


def test_male_voice_and_forced_language(fake_tts):
    _, segs = asyncio.run(VoiceEngine().speak("Hello నమస్కారం", VoiceOptions(gender="male", language="te")))
    assert len(segs) == 1 and segs[0]["voice"] == "te-IN-MohanNeural"


def test_tts_cache(fake_tts):
    eng = VoiceEngine()
    asyncio.run(eng.speak("Hello there"))
    asyncio.run(eng.speak("Hello there"))
    assert len(fake_tts.calls) == 1


def test_produce_timeline_and_mix(fake_tts):
    eng = VoiceEngine()
    prod = asyncio.run(eng.produce("[bg:rain] కథ మొదలు. [sfx:thunder 1s] [pause:300ms] The end. [bg:off]"))
    kinds = [t.kind for t in prod.timeline]
    assert kinds.count("speech") == 2 and "sfx" in kinds and "pause" in kinds and "bg" in kinds
    bg = next(t for t in prod.timeline if t.kind == "bg")
    assert bg.start == 0 and bg.duration > 1.0
    assert np.isfinite(prod.audio).all() and np.abs(prod.audio).max() <= 1.0
    assert prod.duration > 1.3
    assert prod.transcript == "కథ మొదలు. The end."


def test_background_runs_to_end_when_not_closed(fake_tts):
    prod = asyncio.run(VoiceEngine().produce("[bg:wind] Hello world"))
    bg = next(t for t in prod.timeline if t.kind == "bg")
    assert abs(bg.start + bg.duration - prod.duration) < 1 / SAMPLE_RATE * 10
