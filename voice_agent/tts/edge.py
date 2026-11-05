"""Microsoft Edge neural TTS (free, no API key) - natural Telugu & English voices."""

from __future__ import annotations

import numpy as np

from ..audio import AudioError, decode
from .base import Prosody, TTSProvider, VoiceProfile


def _signed(value: int, unit: str) -> str:
    return f"{'+' if value >= 0 else ''}{int(value)}{unit}"


class EdgeTTSProvider(TTSProvider):
    name = "edge"

    def is_available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False
        return True

    async def synthesize(self, text: str, voice: VoiceProfile, prosody: Prosody) -> np.ndarray:
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            voice.id,
            rate=_signed(prosody.rate, "%"),
            pitch=_signed(prosody.pitch, "Hz"),
            volume=_signed(prosody.volume, "%"),
        )
        chunks = bytearray()
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
        except Exception as exc:  # network / service errors
            raise AudioError(f"edge-tts failed for voice {voice.id}: {exc}") from exc
        if not chunks:
            raise AudioError("edge-tts returned no audio")
        return decode(bytes(chunks))
