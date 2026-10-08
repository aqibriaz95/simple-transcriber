import subprocess
from pathlib import Path

import pytest


def _generate(target: Path, source: str, channels: int = 2) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", source, "-ac", str(channels), str(target)],
        check=True,
    )
    return target


def generate_tone(target: Path, seconds: float = 1.0, sample_rate: int = 44100) -> Path:
    return _generate(target, f"sine=frequency=440:sample_rate={sample_rate}:duration={seconds}")


def generate_tone_with_pauses(target: Path, seconds: float, period: float = 5.0, pause: float = 1.0) -> Path:
    expression = f"sin(2*PI*440*t)*lt(mod(t\\,{period})\\,{period - pause})"
    return _generate(target, f"aevalsrc={expression}:s=16000:d={seconds}", channels=1)


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    return generate_tone(tmp_path / "tone.wav")


@pytest.fixture
def mp3_file(tmp_path: Path) -> Path:
    return generate_tone(tmp_path / "tone.mp3")


@pytest.fixture
def long_wav_file(tmp_path: Path) -> Path:
    return generate_tone_with_pauses(tmp_path / "long.wav", seconds=32)
