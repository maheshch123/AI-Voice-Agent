import numpy as np
import pytest

from voice_agent.config import SAMPLE_RATE
from voice_agent.tts.base import TTSProvider
from voice_agent.tts.registry import register_provider


class FakeTTS(TTSProvider):
    """Deterministic offline TTS: a tone whose length tracks the text length."""

    name = "fake"

    def __init__(self):
        self.calls = []

    async def synthesize(self, text, voice, prosody):
        self.calls.append((text, voice.id))
        n = int(SAMPLE_RATE * (0.05 + 0.02 * len(text)))
        t = np.arange(n) / SAMPLE_RATE
        freq = 220 if voice.language == "te" else 330
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def fake_tts(monkeypatch):
    fake = FakeTTS()
    register_provider(fake)
    monkeypatch.setenv("VOICE_AGENT_TTS_PROVIDER", "fake")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    return fake
