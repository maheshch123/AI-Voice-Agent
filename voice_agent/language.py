"""Language detection and code-mixed (Telugu + English) segmentation.

Telugu is written in its own Unicode block (U+0C00-U+0C7F), so script-based
detection is exact and instant - no model needed. Mixed "Tenglish" sentences
such as "నేను office కి వెళ్తున్నాను" are split into runs so each run can be
spoken by a native voice for that language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TELUGU = "te"
ENGLISH = "en"
LANGUAGES = (TELUGU, ENGLISH)

_TELUGU_RANGE = (0x0C00, 0x0C7F)


def char_language(ch: str) -> str | None:
    """Return the language a character belongs to, or None if neutral (space, digit, punctuation)."""
    cp = ord(ch)
    if _TELUGU_RANGE[0] <= cp <= _TELUGU_RANGE[1]:
        return TELUGU
    if ch.isalpha():
        # Any other alphabetic script is spoken by the English voice.
        return ENGLISH
    return None


def detect_language(text: str) -> str:
    """Dominant language of ``text`` by letter count (defaults to English)."""
    te = sum(1 for c in text if char_language(c) == TELUGU)
    en = sum(1 for c in text if char_language(c) == ENGLISH)
    return TELUGU if te > en else ENGLISH


@dataclass(frozen=True)
class Segment:
    text: str
    language: str


def segment_text(text: str, min_english_words: int = 2) -> list[Segment]:
    """Split ``text`` into language runs.

    Neutral characters stick to the current run. A lone English word inside
    Telugu ("నేను office కి వెళ్తాను") stays with the Telugu voice - that's how
    Telugu speakers say it, and it avoids flipping voices mid-sentence.
    """
    runs: list[list] = []  # [language, text]
    for ch in text:
        lang = char_language(ch)
        if not runs:
            runs.append([lang, ch])
        elif lang is None or runs[-1][0] in (None, lang):
            if runs[-1][0] is None:
                runs[-1][0] = lang
            runs[-1][1] += ch
        else:
            runs.append([lang, ch])

    # Merge runs. Telugu voices read short Latin tokens ("TV", "OK") fine, but
    # English voices cannot read Telugu script at all - so only short *English*
    # runs are folded into a neighbouring Telugu run, never the other way round.
    def short(chunk: str) -> bool:
        words = [w for w in re.split(r"[^A-Za-z]+", chunk) if w]
        return len(words) < min_english_words

    merged: list[list] = []
    for lang, chunk in runs:
        if merged and merged[-1][0] in (lang, None):
            merged[-1][0] = lang
            merged[-1][1] += chunk
        elif merged and lang == ENGLISH and short(chunk):
            merged[-1][1] += chunk  # short English token inside Telugu
        elif merged and merged[-1][0] == ENGLISH and len(merged) == 1 and short(merged[-1][1]) and lang == TELUGU:
            merged[-1] = [TELUGU, merged[-1][1] + chunk]  # leading short English token
        else:
            merged.append([lang, chunk])

    fallback = detect_language(text)
    out = []
    for lang, chunk in merged:
        if chunk.strip():
            out.append(Segment(text=re.sub(r"\s+", " ", chunk).strip(), language=lang or fallback))
    return out
