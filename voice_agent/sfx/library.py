"""Sound-effect front door: name or free-text prompt (English or Telugu) -> audio.

Resolution order:
1. exact effect name (``rain``) or Telugu label (``వర్షం``)
2. generative ElevenLabs sound effects, when ``ELEVENLABS_API_KEY`` is set and
   the caller allows it (best for arbitrary prompts like "a dog barking twice")
3. keyword match against the procedural library (fully offline)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
import numpy as np

from ..audio import AudioError, decode, fade, normalize
from ..config import get_settings
from .procedural import EFFECTS, render


@dataclass(frozen=True)
class EffectMatch:
    name: str
    score: float


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^\wఀ-౿]+", text.lower()) if t]


def match_effect(prompt: str) -> EffectMatch | None:
    """Best procedural effect for a prompt, by name, Telugu label or keyword."""
    p = prompt.strip().lower()
    if not p:
        return None
    key = p.replace(" ", "_")
    if key in EFFECTS:
        return EffectMatch(key, 1.0)
    for spec in EFFECTS.values():
        if p == spec.label_te:
            return EffectMatch(spec.name, 1.0)

    tokens = set(_tokens(p))
    best: EffectMatch | None = None
    for spec in EFFECTS.values():
        score = 0.0
        for kw in spec.keywords:
            kw_l = kw.lower()
            if " " in kw_l and kw_l in p:
                score += 2.0  # multi-word phrase match is strong
            elif kw_l in tokens:
                score += 1.0
            elif len(kw_l) >= 4 and any(t.startswith(kw_l) or kw_l.startswith(t) and len(t) >= 4 for t in tokens):
                score += 0.5  # inflections: "raining", "వర్షంలో"
        if score > 0 and (best is None or score > best.score):
            best = EffectMatch(spec.name, score)
    return best


class SoundEffectLibrary:
    def __init__(self) -> None:
        self._cache: dict[tuple, np.ndarray] = {}

    def catalogue(self) -> list[dict]:
        return [
            {"name": s.name, "label_te": s.label_te, "category": s.category,
             "default_duration": s.default_duration, "loopable": s.loopable}
            for s in EFFECTS.values()
        ]

    async def generate(self, prompt: str, duration: float | None = None, provider: str = "auto") -> tuple[np.ndarray, dict]:
        """Return (audio, info). ``provider``: auto | procedural | elevenlabs."""
        match = match_effect(prompt)
        exact = match is not None and match.score >= 1.0 and (
            prompt.strip().lower().replace(" ", "_") in EFFECTS or prompt.strip() == EFFECTS[match.name].label_te
        )
        use_eleven = provider == "elevenlabs" or (
            provider == "auto" and not exact and get_settings().elevenlabs_api_key is not None
        )
        if use_eleven:
            try:
                audio = await self._elevenlabs(prompt, duration)
                return audio, {"source": "elevenlabs", "prompt": prompt}
            except AudioError:
                if provider == "elevenlabs":
                    raise
        if match is None:
            raise LookupError(f"No sound effect matches '{prompt}'.")
        key = (match.name, round(duration or 0, 2))
        if key not in self._cache:
            self._cache[key] = render(match.name, duration)
        return self._cache[key], {"source": "procedural", "effect": match.name, "score": match.score}

    async def _elevenlabs(self, prompt: str, duration: float | None) -> np.ndarray:
        key = get_settings().elevenlabs_api_key
        if not key:
            raise AudioError("ELEVENLABS_API_KEY is not set")
        body: dict = {"text": prompt, "prompt_influence": 0.4}
        if duration:
            body["duration_seconds"] = float(np.clip(duration, 0.5, 22))
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                "https://api.elevenlabs.io/v1/sound-generation",
                headers={"xi-api-key": key, "accept": "audio/mpeg"},
                json=body,
            )
        if resp.status_code != 200:
            raise AudioError(f"ElevenLabs sound effect error {resp.status_code}: {resp.text[:300]}")
        return fade(normalize(decode(resp.content), 0.9))
