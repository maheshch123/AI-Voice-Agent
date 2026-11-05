"""Common TTS interface and the bilingual voice catalogue."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoiceProfile:
    id: str  # provider voice id
    name: str
    language: str  # "te" | "en"
    gender: str  # "female" | "male"
    locale: str
    provider: str = "edge"


# Microsoft neural voices (served by edge-tts). Indian-English voices are the
# default for English so code-mixed sentences keep a consistent accent.
VOICES: list[VoiceProfile] = [
    VoiceProfile("te-IN-ShrutiNeural", "Shruti", "te", "female", "te-IN"),
    VoiceProfile("te-IN-MohanNeural", "Mohan", "te", "male", "te-IN"),
    VoiceProfile("en-IN-NeerjaNeural", "Neerja", "en", "female", "en-IN"),
    VoiceProfile("en-IN-PrabhatNeural", "Prabhat", "en", "male", "en-IN"),
    VoiceProfile("en-US-AriaNeural", "Aria", "en", "female", "en-US"),
    VoiceProfile("en-US-GuyNeural", "Guy", "en", "male", "en-US"),
    VoiceProfile("en-GB-SoniaNeural", "Sonia", "en", "female", "en-GB"),
]


def pick_voice(language: str, gender: str = "female", accent: str = "IN") -> VoiceProfile:
    """Choose the best catalogue voice for a language/gender (English accent: IN | US | GB)."""
    candidates = [v for v in VOICES if v.language == language]
    for v in candidates:
        if v.gender == gender and (language == "te" or v.locale.endswith(accent)):
            return v
    for v in candidates:
        if v.gender == gender:
            return v
    return candidates[0]


@dataclass(frozen=True)
class Prosody:
    rate: int = 0  # percent, -50..+100
    pitch: int = 0  # Hz offset, -50..+50
    volume: int = 0  # percent, -50..+50


class TTSProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def synthesize(self, text: str, voice: VoiceProfile, prosody: Prosody) -> np.ndarray:
        """Return mono float32 audio at ``config.SAMPLE_RATE``."""

    def is_available(self) -> bool:
        return True
