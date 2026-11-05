import asyncio

import numpy as np
import pytest

from voice_agent.config import SAMPLE_RATE
from voice_agent.sfx import EFFECTS, SoundEffectLibrary, match_effect, render


@pytest.mark.parametrize("name", list(EFFECTS))
def test_every_effect_renders_cleanly(name):
    audio = render(name, 1.0)
    assert audio.dtype == np.float32
    assert len(audio) == SAMPLE_RATE
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 1.0
    assert np.abs(audio).max() > 0.05, "effect is silent"


def test_render_is_deterministic():
    assert np.array_equal(render("rain", 0.5), render("rain", 0.5))


@pytest.mark.parametrize("prompt,expected", [
    ("thunder", "thunder"),
    ("temple bell", "temple_bell"),
    ("heavy rain storm at night", "rain"),
    ("వర్షం", "rain"),
    ("గుడి గంట", "temple_bell"),
    ("చప్పట్లు", "applause"),
    ("crowd clapping", "applause"),
    ("dappu beat for the festival", "drum"),
    ("శంఖం", "conch"),
])
def test_prompt_matching(prompt, expected):
    assert match_effect(prompt).name == expected


def test_no_match_raises(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(LookupError):
        asyncio.run(SoundEffectLibrary().generate("zzqx"))
