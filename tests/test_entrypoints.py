import json

from fastapi.testclient import TestClient

from transcription_pipeline import AudioConfig, MockTranscriber
from transcription_pipeline.api import create_app
from transcription_pipeline.cli import main


def test_cli_prints_json_transcription(wav_file, capsys):
    assert main([str(wav_file), "--mock"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["model"] == "mock"
    assert output["segments"][0]["text"].startswith("Hello")


def test_cli_writes_srt_file(mp3_file, tmp_path):
    target = tmp_path / "out.srt"
    assert main([str(mp3_file), "--mock", "--format", "srt", "--output", str(target)]) == 0
    assert target.read_text().startswith("1\n00:00:00,000 -->")


def test_cli_returns_2_for_invalid_input(tmp_path, capsys):
    assert main([str(tmp_path / "missing.mp3"), "--mock"]) == 2
    assert "not found" in capsys.readouterr().err


def test_api_transcribes_mp3_upload(mp3_file):
    client = TestClient(create_app(MockTranscriber()))
    with mp3_file.open("rb") as handle:
        response = client.post("/transcriptions", files={"file": ("meeting.mp3", handle, "audio/mpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "meeting.mp3"
    assert body["audio"]["codec"] == "mp3"
    assert len(body["segments"]) == 4


def test_api_rejects_unsupported_type():
    client = TestClient(create_app(MockTranscriber()))
    response = client.post("/transcriptions", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_api_rejects_corrupt_audio():
    client = TestClient(create_app(MockTranscriber()))
    response = client.post("/transcriptions", files={"file": ("broken.wav", b"not audio", "audio/wav")})
    assert response.status_code == 400


def test_api_rejects_oversized_upload(wav_file):
    client = TestClient(create_app(MockTranscriber(), AudioConfig(max_file_size_bytes=1024)))
    with wav_file.open("rb") as handle:
        response = client.post("/transcriptions", files={"file": ("big.wav", handle, "audio/wav")})
    assert response.status_code == 413
