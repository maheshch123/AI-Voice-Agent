"""Command line interface.

    python -m voice_agent say "నమస్కారం! Welcome to Vani." -o hello.mp3
    python -m voice_agent produce story.txt -o story.mp3
    python -m voice_agent sfx "thunder" --duration 4 -o thunder.wav
    python -m voice_agent sfx --list
    python -m voice_agent serve
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .audio import encode
from .engine import VoiceEngine, VoiceOptions
from .sfx import EFFECTS


def _write(samples, path: str) -> None:
    fmt = "mp3" if path.lower().endswith(".mp3") else "wav"
    data, _ = encode(samples, fmt)
    Path(path).write_bytes(data)
    print(f"wrote {path} ({len(samples) / 24000:.1f}s)")


def _opts(a) -> VoiceOptions:
    return VoiceOptions(gender=a.gender, language=a.lang, english_accent=a.accent, rate=a.rate, pitch=a.pitch,
                        provider=a.provider)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="voice_agent", description="Vani: Telugu & English AI voice agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    def voice_args(sp):
        sp.add_argument("--gender", choices=["female", "male"], default="female")
        sp.add_argument("--lang", choices=["auto", "te", "en"], default="auto")
        sp.add_argument("--accent", choices=["IN", "US", "GB"], default="IN")
        sp.add_argument("--rate", type=int, default=0, help="speed change in percent")
        sp.add_argument("--pitch", type=int, default=0, help="pitch change in Hz")
        sp.add_argument("--provider", choices=["edge", "elevenlabs"], default=None)
        sp.add_argument("-o", "--output", default="output.mp3")

    say = sub.add_parser("say", help="speak text (Telugu, English or mixed)")
    say.add_argument("text")
    voice_args(say)

    prod = sub.add_parser("produce", help="render a script file with [sfx:..] [bg:..] [pause:..] tags")
    prod.add_argument("script", help="path to script file, or - for stdin")
    voice_args(prod)

    sfx = sub.add_parser("sfx", help="generate a sound effect")
    sfx.add_argument("prompt", nargs="?")
    sfx.add_argument("--duration", type=float)
    sfx.add_argument("--provider", choices=["auto", "procedural", "elevenlabs"], default="auto")
    sfx.add_argument("--list", action="store_true", help="list built-in effects")
    sfx.add_argument("-o", "--output", default="sfx.wav")

    serve = sub.add_parser("serve", help="run the web studio + API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    engine = VoiceEngine()

    if a.cmd == "say":
        samples, segs = asyncio.run(engine.speak(a.text, _opts(a)))
        for s in segs:
            print(f"  [{s['language']}] {s['voice']}: {s['text']}")
        _write(samples, a.output)
    elif a.cmd == "produce":
        script = sys.stdin.read() if a.script == "-" else Path(a.script).read_text(encoding="utf-8")
        prodn = asyncio.run(engine.produce(script, _opts(a)))
        for item in prodn.timeline:
            print(f"  {item.start:6.2f}s  {item.kind:6s} {item.duration:5.2f}s  {item.detail.get('text') or item.detail.get('effect') or ''}")
        _write(prodn.audio, a.output)
    elif a.cmd == "sfx":
        if a.list or not a.prompt:
            for e in EFFECTS.values():
                print(f"  {e.name:14s} {e.label_te:16s} {e.category:10s} {e.default_duration:4.1f}s")
            return 0
        samples, info = asyncio.run(engine.sfx.generate(a.prompt, a.duration, a.provider))
        print(f"  {info}")
        _write(samples, a.output)
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("voice_agent.server:app", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
