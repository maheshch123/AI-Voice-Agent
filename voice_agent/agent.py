"""Conversational brain for the voice agent.

With an Anthropic credential available it uses Claude, which replies in the
user's language (Telugu, English or natural code-mixed "Tenglish") and can
decorate its spoken reply with sound-effect tags that the engine renders.
Without a credential it falls back to a small offline responder so the rest of
the app keeps working.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import get_settings
from .language import TELUGU, detect_language
from .sfx.procedural import EFFECTS

log = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are Vani (వాణి), a warm, friendly voice assistant fluent in Telugu and English.

Everything you write is converted to speech, so:
- Reply in the language the user used. Telugu input -> Telugu script (not transliteration). English -> English. If the user mixes both, reply in the same natural mix.
- Keep replies short and conversational: usually 1-3 sentences. No markdown, lists, emojis, URLs or code - they cannot be spoken.
- Write numbers, dates and abbreviations the way they should be read aloud.

You can add sound effects to make replies expressive. Use them sparingly (at most two per reply, only when they genuinely add something, e.g. celebrating, storytelling, greetings):
- [sfx:NAME] plays an effect inline, e.g. "అభినందనలు! [sfx:applause]"
- [bg:NAME] ... [bg:off] plays a soft ambience under your words (good for stories)
- [pause:500ms] adds a dramatic pause
Available NAME values: {", ".join(EFFECTS)}."""


@dataclass
class AgentReply:
    text: str
    language: str
    source: str  # "claude" | "offline"


class VoiceAgent:
    def __init__(self) -> None:
        self._client = None

    @property
    def enabled(self) -> bool:
        return get_settings().anthropic_enabled

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def reply(self, message: str, history: list[dict] | None = None) -> AgentReply:
        lang = detect_language(message)
        if not self.enabled:
            return AgentReply(offline_reply(message, lang), lang, "offline")

        import anthropic

        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in (history or [])[-20:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        messages.append({"role": "user", "content": message})
        try:
            response = await self._get_client().beta.messages.create(
                model=get_settings().agent_model,
                max_tokens=4096,  # replies are short spoken sentences
                system=SYSTEM_PROMPT,
                output_config={"effort": "low"},  # snappy conversational latency
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=messages,
            )
        except anthropic.AuthenticationError:
            log.warning("Anthropic credential rejected; using offline responder")
            return AgentReply(offline_reply(message, lang), lang, "offline")
        except anthropic.RateLimitError:
            return AgentReply(_busy(lang), lang, "offline")
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            log.warning("Claude request failed: %s", exc)
            return AgentReply(_busy(lang), lang, "offline")

        if response.stop_reason == "refusal":
            text = ("క్షమించండి, దానికి నేను సహాయం చేయలేను." if lang == TELUGU
                    else "Sorry, I can't help with that one.")
            return AgentReply(text, lang, "claude")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return AgentReply(text or _busy(lang), detect_language(text) if text else lang, "claude")


def _busy(lang: str) -> str:
    if lang == TELUGU:
        return "క్షమించండి, ఇప్పుడు నేను కనెక్ట్ కాలేకపోతున్నాను. కొద్దిసేపటి తర్వాత మళ్ళీ ప్రయత్నించండి."
    return "Sorry, I can't reach my brain right now. Please try again in a moment."


_OFFLINE_RULES = [
    (("నమస్కారం", "నమస్తే", "హలో", "hello", "hi", "hey", "namaste", "namaskaram"),
     "[sfx:chime] నమస్కారం! నేను వాణి, మీ వాయిస్ అసిస్టెంట్. మీకు ఎలా సహాయం చేయగలను?",
     "[sfx:chime] Hello! I'm Vani, your voice assistant. How can I help you today?"),
    (("ధన్యవాదాలు", "థాంక్స్", "thank", "thanks"),
     "మీకు స్వాగతం! [sfx:success]", "You're welcome! [sfx:success]"),
    (("పేరు", "name", "who are you", "ఎవరు"),
     "నా పేరు వాణి. నేను తెలుగు మరియు ఇంగ్లీష్ రెండూ మాట్లాడగలను.",
     "My name is Vani. I can speak both Telugu and English."),
    (("వర్షం", "rain", "కథ", "story"),
     "[bg:rain] ఒకప్పుడు, ఒక చిన్న గ్రామంలో వర్షం కురుస్తున్న రాత్రి... [sfx:thunder] [bg:off]",
     "[bg:rain] Once upon a time, on a rainy night in a small village... [sfx:thunder] [bg:off]"),
    (("పండుగ", "festival", "celebrate", "సంబరం"),
     "[sfx:drum] పండుగ శుభాకాంక్షలు!", "[sfx:drum] Happy festival! Let's celebrate!"),
]


def offline_reply(message: str, lang: str) -> str:
    low = message.lower()
    for keys, te, en in _OFFLINE_RULES:
        if any(k in low for k in keys):
            return te if lang == TELUGU else en
    if lang == TELUGU:
        return ("నేను ప్రస్తుతం ఆఫ్‌లైన్ మోడ్‌లో ఉన్నాను. పూర్తి సంభాషణ కోసం ANTHROPIC_API_KEY ని సెట్ చేయండి. "
                "అయినా, మీరు చెప్పింది నేను చదివి వినిపిస్తాను: " + message)
    return ("I'm in offline mode right now. Set ANTHROPIC_API_KEY for full conversations. "
            "Meanwhile, here's what you said: " + message)
