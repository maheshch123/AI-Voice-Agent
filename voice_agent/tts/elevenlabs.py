"""ElevenLabs TTS (optional, needs ELEVENLABS_API_KEY).

``eleven_v3`` is multilingual and covers Telugu and English with one voice, so
the same voice is used for both languages. Set ELEVENLABS_VOICE_FEMALE /
ELEVENLABS_VOICE_MALE to use your own (or cloned) voices.
"""

from __future__ import annotations

import os

import httpx
import numpy as np

from ..audio import AudioError, decode
from ..config import get_settings
from .base import Prosody, TTSProvider, VoiceProfile

API = "https://api.elevenlabs.io/v1"
# Public ElevenLabs library voices ("Rachel", "Adam").
_DEFAULT_VOICES = {"female": "21m00Tcm4TlvDq8ikWAM", "male": "pNInz6obpgDQGcFmaJgB"}


class ElevenLabsTTSProvider(TTSProvider):
    name = "elevenlabs"

    def is_available(self) -> bool:
        return bool(get_settings().elevenlabs_api_key)

    def _voice_id(self, gender: str) -> str:
        env = os.getenv(f"ELEVENLABS_VOICE_{gender.upper()}")
        return env or _DEFAULT_VOICES.get(gender, _DEFAULT_VOICES["female"])

    async def synthesize(self, text: str, voice: VoiceProfile, prosody: Prosody) -> np.ndarray:
        settings = get_settings()
        if not settings.elevenlabs_api_key:
            raise AudioError("ELEVENLABS_API_KEY is not set")
        body = {
            "text": text,
            "model_id": settings.elevenlabs_tts_model,
            "language_code": voice.language,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": max(0.7, min(1.2, 1 + prosody.rate / 100))},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{API}/text-to-speech/{self._voice_id(voice.gender)}",
                headers={"xi-api-key": settings.elevenlabs_api_key, "accept": "audio/mpeg"},
                json=body,
            )
        if resp.status_code != 200:
            raise AudioError(f"ElevenLabs TTS error {resp.status_code}: {resp.text[:300]}")
        return decode(resp.content)
