from __future__ import annotations

from .base import TTSProvider
from .edge import EdgeTTSProvider
from .elevenlabs import ElevenLabsTTSProvider

_PROVIDERS: dict[str, TTSProvider] = {
    "edge": EdgeTTSProvider(),
    "elevenlabs": ElevenLabsTTSProvider(),
}


def register_provider(provider: TTSProvider) -> None:
    """Register a custom provider (also used by tests to inject a fake)."""
    _PROVIDERS[provider.name] = provider


def get_provider(name: str) -> TTSProvider:
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown TTS provider '{name}'. Available: {sorted(_PROVIDERS)}") from None


def available_providers() -> dict[str, bool]:
    return {name: p.is_available() for name, p in _PROVIDERS.items()}
