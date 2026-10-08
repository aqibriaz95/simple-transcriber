import json
import shutil
import subprocess
from pathlib import Path

import pytest

from transcription_pipeline import AudioConfig, AudioLoader, AudioValidationError


def _probe(path: Path) -> dict:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(output)["streams"][0]


@pytest.mark.parametrize("fixture_name, expected_format", [("wav_file", "wav"), ("mp3_file", "mp3")])
def test_load_accepts_supported_formats(request, tmp_path, fixture_name, expected_format):
    source = request.getfixturevalue(fixture_name)
    with AudioLoader(work_dir=tmp_path / "work") as loader:
        audio = loader.load(source)

    assert audio.metadata.format_name == expected_format
    assert audio.metadata.channels == 2
    assert audio.metadata.duration_seconds == pytest.approx(1.0, abs=0.1)
    assert audio.normalized_path.is_file()

    stream = _probe(audio.normalized_path)
    assert stream["codec_name"] == "pcm_s16le"
    assert int(stream["sample_rate"]) == 16000
    assert stream["channels"] == 1


def test_temporary_work_dir_is_removed_on_exit(wav_file):
    with AudioLoader() as loader:
        audio = loader.load(wav_file)
        work_dir = loader.work_dir
        assert audio.normalized_path.is_file()
    assert not work_dir.exists()


def test_missing_file_is_rejected(tmp_path):
    with AudioLoader() as loader, pytest.raises(AudioValidationError, match="not found"):
        loader.load(tmp_path / "missing.wav")


def test_unsupported_extension_is_rejected(tmp_path):
    source = tmp_path / "speech.flac"
    source.write_bytes(b"fLaC")
    with AudioLoader() as loader, pytest.raises(AudioValidationError, match="Unsupported"):
        loader.load(source)


def test_empty_file_is_rejected(tmp_path):
    source = tmp_path / "empty.wav"
    source.touch()
    with AudioLoader() as loader, pytest.raises(AudioValidationError, match="empty"):
        loader.load(source)


def test_non_audio_content_is_rejected(tmp_path):
    source = tmp_path / "notes.wav"
    source.write_text("this is not audio")
    with AudioLoader() as loader, pytest.raises(AudioValidationError):
        loader.load(source)


def test_mislabelled_extension_is_rejected(wav_file, tmp_path):
    disguised = tmp_path / "disguised.mp3"
    shutil.copy(wav_file, disguised)
    with AudioLoader() as loader, pytest.raises(AudioValidationError, match="contains 'wav'"):
        loader.load(disguised)


def test_file_size_limit_is_enforced(wav_file):
    with AudioLoader(AudioConfig(max_file_size_bytes=100)) as loader:
        with pytest.raises(AudioValidationError, match="limit"):
            loader.load(wav_file)


def test_duration_limit_is_enforced(wav_file):
    with AudioLoader(AudioConfig(max_duration_seconds=0.5)) as loader:
        with pytest.raises(AudioValidationError, match="long"):
            loader.load(wav_file)
