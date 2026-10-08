import logging
import re
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg
from .audio import AudioInput
from .config import ChunkingConfig
from .errors import AudioProcessingError

logger = logging.getLogger(__name__)

SILENCE_PATTERN = re.compile(r"silence_(start|end): (-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    offset: float
    duration: float


def plan_cuts(
    duration: float,
    silence_midpoints: list[float],
    chunk_seconds: float,
    search_window_seconds: float,
) -> list[float]:
    cuts = []
    position = 0.0
    while duration - position > chunk_seconds:
        target = position + chunk_seconds
        candidates = [
            point
            for point in silence_midpoints
            if position < point <= target and point >= target - search_window_seconds
        ]
        cut = max(candidates) if candidates else target
        cuts.append(round(cut, 3))
        position = cut
    return cuts


def parse_silences(ffmpeg_log: str) -> list[float]:
    midpoints = []
    start = None
    for kind, value in SILENCE_PATTERN.findall(ffmpeg_log):
        if kind == "start":
            start = max(float(value), 0.0)
        elif start is not None:
            midpoints.append((start + float(value)) / 2)
            start = None
    return midpoints


class AudioChunker:
    def __init__(self, work_dir: Path, config: ChunkingConfig | None = None):
        self._config = config or ChunkingConfig()
        self._work_dir = work_dir
        self._ffmpeg = ffmpeg.find_binary("ffmpeg")

    def split(self, audio: AudioInput) -> list[AudioChunk]:
        duration = audio.metadata.duration_seconds
        if duration <= self._config.chunk_seconds:
            return [AudioChunk(audio.normalized_path, 0.0, duration)]

        silences = self._detect_silences(audio.normalized_path)
        cuts = plan_cuts(
            duration,
            silences,
            self._config.chunk_seconds,
            self._config.search_window_seconds,
        )
        bounds = list(zip([0.0, *cuts], [*cuts, duration]))
        logger.info("Split %.1fs of audio into %d chunks at %s", duration, len(bounds), cuts)
        return [
            self._extract(audio.normalized_path, index, start, end)
            for index, (start, end) in enumerate(bounds)
        ]

    def _detect_silences(self, path: Path) -> list[float]:
        command = [
            self._ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-i", str(path),
            "-af",
            f"silencedetect=noise={self._config.silence_threshold_db}dB:d={self._config.min_silence_seconds}",
            "-f", "null",
            "-",
        ]
        result = ffmpeg.run(command, self._config.detect_timeout_seconds)
        if result.returncode != 0:
            raise AudioProcessingError(f"Silence detection failed: {result.stderr.strip()[-500:]}")
        return parse_silences(result.stderr)

    def _extract(self, path: Path, index: int, start: float, end: float) -> AudioChunk:
        target = self._work_dir / f"{path.stem}_chunk{index:04d}.wav"
        command = [
            self._ffmpeg,
            "-nostdin",
            "-y",
            "-v", "error",
            "-ss", f"{start:.3f}",
            "-i", str(path),
            "-t", f"{end - start:.3f}",
            "-c", "copy",
            str(target),
        ]
        result = ffmpeg.run(command, self._config.extract_timeout_seconds)
        if result.returncode != 0 or not target.is_file():
            raise AudioProcessingError(f"Could not extract chunk {index}: {result.stderr.strip()}")
        return AudioChunk(target, start, end - start)
