"""FastAPI server: REST API + the web studio UI."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .agent import VoiceAgent
from .audio import AudioError, encode, ffmpeg_available
from .config import SAMPLE_RATE, get_settings
from .engine import VoiceEngine, VoiceOptions
from .script import strip_tags
from .sfx import EFFECTS
from .tts import VOICES, available_providers

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Vani - Telugu & English AI Voice Agent", version=__version__)
engine = VoiceEngine()
agent = VoiceAgent()


class VoiceParams(BaseModel):
    gender: Literal["female", "male"] = "female"
    language: Literal["auto", "te", "en"] = "auto"
    english_accent: Literal["IN", "US", "GB"] = "IN"
    rate: int = Field(0, ge=-50, le=100)
    pitch: int = Field(0, ge=-50, le=50)
    volume: int = Field(0, ge=-50, le=50)
    provider: Literal["edge", "elevenlabs"] | None = None
    sfx_provider: Literal["auto", "procedural", "elevenlabs"] = "auto"
    format: Literal["wav", "mp3"] = "mp3"
    download: bool = False  # true -> raw audio bytes instead of JSON

    def options(self) -> VoiceOptions:
        return VoiceOptions(gender=self.gender, language=self.language, english_accent=self.english_accent,
                            rate=self.rate, pitch=self.pitch, volume=self.volume, provider=self.provider,
                            sfx_provider=self.sfx_provider)


class TTSRequest(VoiceParams):
    text: str = Field(..., min_length=1, max_length=5000)


class ProduceRequest(VoiceParams):
    script: str = Field(..., min_length=1, max_length=10000)


class SFXRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=300)
    duration: float | None = Field(None, gt=0, le=30)
    provider: Literal["auto", "procedural", "elevenlabs"] = "auto"
    format: Literal["wav", "mp3"] = "mp3"
    download: bool = False


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(VoiceParams):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatTurn] = []
    speak: bool = True


def _fmt(requested: str) -> str:
    return requested if requested != "mp3" or ffmpeg_available() else "wav"


def _audio_payload(samples, fmt: str, download: bool, **extra):
    data, mime = encode(samples, _fmt(fmt))
    if download:
        ext = "mp3" if mime == "audio/mpeg" else "wav"
        return Response(data, media_type=mime, headers={"Content-Disposition": f'attachment; filename="vani.{ext}"'})
    return {"audio": base64.b64encode(data).decode(), "mime": mime,
            "duration": round(len(samples) / SAMPLE_RATE, 3), **extra}


@app.get("/api/health")
def health():
    s = get_settings()
    return {"status": "ok", "version": __version__, "tts_providers": available_providers(),
            "default_tts_provider": s.tts_provider, "agent": "claude" if agent.enabled else "offline",
            "elevenlabs": bool(s.elevenlabs_api_key), "ffmpeg": ffmpeg_available()}


@app.get("/api/voices")
def voices():
    return [v.__dict__ for v in VOICES]


@app.get("/api/sfx")
def list_sfx():
    return engine.sfx.catalogue()


@app.post("/api/tts")
async def tts(req: TTSRequest):
    try:
        samples, segments = await engine.speak(req.text, req.options())
    except (AudioError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc
    return _audio_payload(samples, req.format, req.download, segments=segments)


@app.post("/api/sfx")
async def sfx(req: SFXRequest):
    try:
        samples, info = await engine.sfx.generate(req.prompt, req.duration, req.provider)
    except LookupError as exc:
        raise HTTPException(404, f"{exc} Try one of: {', '.join(EFFECTS)}") from exc
    except AudioError as exc:
        raise HTTPException(502, str(exc)) from exc
    return _audio_payload(samples, req.format, req.download, info=info)


@app.post("/api/produce")
async def produce(req: ProduceRequest):
    try:
        prod = await engine.produce(req.script, req.options())
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (AudioError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc
    return _audio_payload(prod.audio, req.format, req.download, transcript=prod.transcript,
                          timeline=[t.__dict__ for t in prod.timeline])


@app.post("/api/agent/chat")
async def chat(req: ChatRequest):
    reply = await agent.reply(req.message, [t.model_dump() for t in req.history])
    result = {"reply": reply.text, "transcript": strip_tags(reply.text), "language": reply.language,
              "source": reply.source}
    if not req.speak:
        return result
    try:
        prod = await engine.produce(reply.text, req.options())
    except (AudioError, LookupError, ValueError) as exc:
        return {**result, "audio_error": str(exc)}
    payload = _audio_payload(prod.audio, req.format, False)
    return {**result, **payload}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:  # pragma: no cover
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Run the Vani voice agent server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run("voice_agent.server:app", host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
