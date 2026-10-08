import shutil
import subprocess
from pathlib import Path

from .errors import AudioProcessingError


def find_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise AudioProcessingError(f"{name} is not installed or not on PATH")
    return binary


def run(command: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError(f"{Path(command[0]).name} timed out after {timeout:.0f}s") from exc
