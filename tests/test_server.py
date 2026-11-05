import base64

from fastapi.testclient import TestClient

from voice_agent.server import app

client = TestClient(app)


def test_health_and_catalogues(fake_tts):
    assert client.get("/api/health").json()["status"] == "ok"
    assert any(v["language"] == "te" for v in client.get("/api/voices").json())
    assert len(client.get("/api/sfx").json()) >= 30


def test_tts_json(fake_tts):
    r = client.post("/api/tts", json={"text": "నమస్కారం Hello world", "format": "wav"})
    assert r.status_code == 200
    body = r.json()
    assert base64.b64decode(body["audio"])[:4] == b"RIFF"
    assert {s["language"] for s in body["segments"]} == {"te", "en"}


def test_tts_download_mp3(fake_tts):
    r = client.post("/api/tts", json={"text": "Hello", "format": "mp3", "download": True})
    assert r.status_code == 200
    assert r.headers["content-type"] in ("audio/mpeg", "audio/wav")


def test_sfx_endpoint(fake_tts):
    r = client.post("/api/sfx", json={"prompt": "వర్షం", "duration": 1, "format": "wav"})
    assert r.status_code == 200 and r.json()["info"]["effect"] == "rain"
    assert client.post("/api/sfx", json={"prompt": "zzqx"}).status_code == 404


def test_produce_endpoint(fake_tts):
    r = client.post("/api/produce", json={"script": "[sfx:chime] Hi [pause:200ms] బాగున్నారా?", "format": "wav"})
    assert r.status_code == 200
    assert [t["kind"] for t in r.json()["timeline"]] == ["sfx", "speech", "pause", "speech"]


def test_agent_offline_mode(fake_tts):
    r = client.post("/api/agent/chat", json={"message": "నమస్కారం", "format": "wav"})
    body = r.json()
    assert body["source"] == "offline" and body["language"] == "te"
    assert "audio" in body and "[sfx" not in body["transcript"]


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "Vani" in r.text
