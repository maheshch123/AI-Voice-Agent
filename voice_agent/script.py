"""Parser for the expressive voice script markup.

A script is plain text with optional inline tags:

    [sfx:thunder]          play a sound effect inline (speech waits for it)
    [sfx:rain 4s]          ... with a duration
    [sfx:వర్షం]            Telugu effect names / prompts work too
    [bg:rain]              start a background ambience bed under everything that follows
    [bg:off]               stop the background bed
    [pause:600ms]          silence (also accepts seconds: [pause:1.5s] or [pause])
    [voice:male]           switch voice gender for following speech (male | female)
    [lang:te] / [lang:en]  force a language for following speech ([lang:auto] to reset)

Everything else is spoken. Unknown tags are spoken as-is so nothing is lost silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TAG = re.compile(r"\[(sfx|bg|pause|voice|lang)(?::\s*([^\]]*))?\]", re.IGNORECASE)
_DURATION = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|sec|seconds?)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Speech:
    text: str
    voice: str | None = None  # "male" | "female" | None (default)
    language: str | None = None  # "te" | "en" | None (auto)


@dataclass(frozen=True)
class Effect:
    prompt: str
    duration: float | None = None


@dataclass(frozen=True)
class Background:
    prompt: str | None  # None = stop


@dataclass(frozen=True)
class Pause:
    seconds: float


Event = Speech | Effect | Background | Pause


def parse_duration(value: str | None, default: float | None = None) -> float | None:
    if not value:
        return default
    m = _DURATION.match(value.strip())
    if not m:
        return default
    num = float(m.group("num"))
    unit = (m.group("unit") or "s").lower()
    return num / 1000 if unit == "ms" else num


def _split_prompt_and_duration(arg: str) -> tuple[str, float | None]:
    parts = arg.strip().rsplit(" ", 1)
    if len(parts) == 2:
        dur = parse_duration(parts[1])
        if dur is not None:
            return parts[0].strip(), dur
    return arg.strip(), None


def parse_script(script: str) -> list[Event]:
    events: list[Event] = []
    voice: str | None = None
    lang: str | None = None
    pos = 0

    def emit_text(chunk: str) -> None:
        if chunk.strip():
            events.append(Speech(text=chunk.strip(), voice=voice, language=lang))

    for m in _TAG.finditer(script):
        emit_text(script[pos : m.start()])
        pos = m.end()
        kind = m.group(1).lower()
        arg = (m.group(2) or "").strip()
        if kind == "sfx" and arg:
            prompt, dur = _split_prompt_and_duration(arg)
            events.append(Effect(prompt=prompt, duration=dur))
        elif kind == "bg":
            if not arg or arg.lower() in ("off", "stop", "none"):
                events.append(Background(prompt=None))
            else:
                events.append(Background(prompt=arg))
        elif kind == "pause":
            events.append(Pause(seconds=min(parse_duration(arg, 0.5) or 0.5, 10.0)))
        elif kind == "voice":
            voice = arg.lower() if arg.lower() in ("male", "female") else None
        elif kind == "lang":
            lang = arg.lower() if arg.lower() in ("te", "en") else None
    emit_text(script[pos:])
    return events


def strip_tags(script: str) -> str:
    """Plain spoken text with all markup removed (for display / transcripts)."""
    return re.sub(r"\s+", " ", _TAG.sub(" ", script)).strip()
