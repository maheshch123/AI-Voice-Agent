"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

SAMPLE_RATE = 24_000  # Hz, mono. Every audio buffer inside the engine uses this rate.


@dataclass(frozen=True)
class Settings:
    tts_provider: str = field(default_factory=lambda: os.getenv("VOICE_AGENT_TTS_PROVIDER", "edge"))
    elevenlabs_api_key: str | None = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY") or None)
    elevenlabs_tts_model: str = field(default_factory=lambda: os.getenv("ELEVENLABS_TTS_MODEL", "eleven_v3"))
    anthropic_enabled: bool = field(
        default_factory=lambda: bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    )
    agent_model: str = field(default_factory=lambda: os.getenv("VOICE_AGENT_MODEL", "claude-opus-5-5"))


def get_settings() -> Settings:
    return Settings()
