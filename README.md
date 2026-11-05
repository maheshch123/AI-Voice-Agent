# Vani (వాణి): Telugu & English AI Voice Agent

Vani is a bilingual voice agent that **speaks Telugu and English** with natural neural voices, **creates sound effects**, mixes them into produced audio, and **holds spoken conversations** in Telugu, English or code-mixed "Tenglish".

| | |
|---|---|
| 🗣️ **Text-to-speech** | Native neural voices: Shruti & Mohan (Telugu), Neerja & Prabhat (Indian English), plus US/UK English. Adjustable speed and pitch. |
| 🔀 **Code-mixed speech** | Detects language from the script, then gives each part of a sentence a native voice: `నమస్కారం! Welcome to Vani` → Telugu voice, then English voice. A lone English word inside Telugu (`నేను office కి వెళ్తాను`) stays in the Telugu voice, the way Telugu speakers say it. |
| 🎧 **Sound effects** | 37 built-in effects synthesized offline (rain, thunder, temple bell / గుడి గంట, conch / శంఖం, dappu drum / డప్పు, applause, heartbeat, magic, explosions…). Request them by name or by an English or Telugu prompt. Optional ElevenLabs generation for any prompt. |
| 🎬 **Audio production** | Script markup puts inline effects, background ambience, pauses, and voice and language switches into one mixed track. Background beds duck automatically under speech. |
| 🤖 **Conversational agent** | Claude-powered. It replies in your language and adds sound effects to its own replies. Speak with the browser mic (`te-IN` / `en-IN`) or type. Falls back to a simple offline responder if no API key is set. |
| 🖥️ **Web studio + API + CLI** | A web UI with Agent, Studio and Sound FX tabs, a REST API, and a command-line tool. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # needs ffmpeg on PATH for MP3 (apt install ffmpeg / brew install ffmpeg)
cp .env.example .env                  # optional: add API keys
python -m voice_agent serve           # open http://127.0.0.1:8000
```

Neural TTS works out of the box. No key is needed, but it needs internet access to Microsoft's speech service. Sound effects work fully offline.

Optional keys (environment variables):

| Variable | Enables |
|---|---|
| `ANTHROPIC_API_KEY` | Claude conversation brain (default model `claude-opus-5-5`, override with `VOICE_AGENT_MODEL`) |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS (`VOICE_AGENT_TTS_PROVIDER=elevenlabs`, model `eleven_v3`) and generative sound effects for prompts that don't match a built-in effect |
| `ELEVENLABS_VOICE_FEMALE` / `ELEVENLABS_VOICE_MALE` | Your own or cloned ElevenLabs voice IDs |

## Script markup

```text
[bg:rain] ఒకప్పుడు, ఒక చిన్న గ్రామంలో, చీకటి రాత్రి వర్షం కురుస్తోంది. [pause:600ms]
[sfx:thunder] అకస్మాత్తుగా, తలుపు దగ్గర చప్పుడు వినిపించింది. [sfx:knock]
[voice:male] ఎవరు అక్కడ? [bg:off]
```

| Tag | Effect |
|---|---|
| `[sfx:NAME]`, `[sfx:NAME 3s]`, `[sfx:గుడి గంట]` | Inline sound effect (name, Telugu label or free-text prompt) |
| `[bg:NAME]` … `[bg:off]` | Looping background ambience, ducked under speech |
| `[pause:500ms]`, `[pause:1.5s]` | Silence |
| `[voice:male]` / `[voice:female]` | Switch voice for the speech that follows |
| `[lang:te]` / `[lang:en]` / `[lang:auto]` | Force a language for the speech that follows |

## CLI

```bash
python -m voice_agent say "నమస్కారం! Welcome to Vani." -o hello.mp3
python -m voice_agent say "Good morning" --gender male --accent US -o gm.wav
python -m voice_agent produce story.txt -o story.mp3      # script with tags ("-" for stdin)
python -m voice_agent sfx "వర్షం" --duration 8 -o rain.wav
python -m voice_agent sfx --list
```

## REST API

| Method | Path | Body |
|---|---|---|
| `GET` | `/api/health` | Provider and agent status |
| `GET` | `/api/voices`, `/api/sfx` | Voice and effect catalogues |
| `POST` | `/api/tts` | `{text, gender?, language?, english_accent?, rate?, pitch?, provider?, format?, download?}` |
| `POST` | `/api/sfx` | `{prompt, duration?, provider?, format?, download?}` |
| `POST` | `/api/produce` | `{script, …voice params}` → audio + timeline |
| `POST` | `/api/agent/chat` | `{message, history?, speak?, …voice params}` → reply text + spoken audio |

By default, audio endpoints return JSON with base64 `audio`, `mime` and `duration`. Pass `"download": true` to get the raw MP3/WAV bytes instead.

```bash
curl -X POST localhost:8000/api/tts -H 'content-type: application/json' \
  -d '{"text":"మీకు స్వాగతం! Welcome!","download":true}' -o welcome.mp3
```

## Architecture

```
voice_agent/
  language.py     Telugu/English detection + code-mixed segmentation (Unicode-script based)
  script.py       Parser for [sfx]/[bg]/[pause]/[voice]/[lang] markup
  engine.py       Concurrent synthesis → timeline → mix (ducking, limiter, TTS cache)
  audio.py        float32 buffers, ffmpeg decode/encode, WAV writer, mixing helpers
  tts/            Provider interface: edge (Microsoft neural, default), elevenlabs
  sfx/            Procedural synthesizer (37 effects) + prompt matcher + ElevenLabs SFX
  agent.py        Claude conversation brain (bilingual, effect-aware) + offline fallback
  server.py       FastAPI app;  web/  single-page studio UI
```

To add a TTS engine, subclass `tts.base.TTSProvider` and call `tts.registry.register_provider()`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake TTS provider, so they run offline.
