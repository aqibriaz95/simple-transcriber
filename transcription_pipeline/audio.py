import json
import logging
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from . import ffmpeg
from .config import AudioConfig
from .errors import AudioProcessingError, AudioValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioMetadata:
    format_name: str
    codec: str
    duration_seconds: float
    sample_rate: int
    channels: int
    size_bytes: int


@dataclass(frozen=True)
class AudioInput:
    source_path: Path
    normalized_path: Path
    metadata: AudioMetadata

    def to_dict(self) -> dict:
        return {
            "source_path": str(self.source_path),
            "normalized_path": str(self.normalized_path),
            "metadata": asdict(self.metadata),
        }


class AudioLoader:
    def __init__(self, config: AudioConfig | None = None, work_dir: str | Path | None = None):
        self._config = config or AudioConfig()
        self._ffmpeg = ffmpeg.find_binary("ffmpeg")
        self._ffprobe = ffmpeg.find_binary("ffprobe")
        self._owns_work_dir = work_dir is None
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="transcription_"))
        self._work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    def __enter__(self) -> "AudioLoader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self._owns_work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def load(self, path: str | Path) -> AudioInput:
        source = Path(path).expanduser().resolve()
        size = self._validate_file(source)
        metadata = self._probe(source, size)
        self._validate_metadata(source, metadata)
        normalized = self._normalize(source)
        logger.info(
            "Loaded %s (%s, %.2fs, %d Hz, %d ch)",
            source.name,
            metadata.codec,
            metadata.duration_seconds,
            metadata.sample_rate,
            metadata.channels,
        )
        return AudioInput(source_path=source, normalized_path=normalized, metadata=metadata)

    def _validate_file(self, source: Path) -> int:
        if not source.is_file():
            raise AudioValidationError(f"File not found: {source.name}")
        extension = source.suffix.lower()
        if extension not in self._config.allowed_formats:
            allowed = ", ".join(sorted(self._config.allowed_formats))
            raise AudioValidationError(f"Unsupported file type '{extension}'. Allowed types: {allowed}")
        size = source.stat().st_size
        if size == 0:
            raise AudioValidationError(f"File is empty: {source.name}")
        if size > self._config.max_file_size_bytes:
            raise AudioValidationError(
                f"File is {size} bytes, the limit is {self._config.max_file_size_bytes} bytes"
            )
        return size

    def _probe(self, source: Path, size: int) -> AudioMetadata:
        command = [
            self._ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(source),
        ]
        result = ffmpeg.run(command, self._config.probe_timeout_seconds)
        if result.returncode != 0:
            reason = result.stderr.strip().replace(str(source), source.name)
            raise AudioValidationError(f"Could not read audio from {source.name}: {reason}")
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AudioProcessingError(f"ffprobe returned invalid output for {source.name}") from exc

        stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        if stream is None:
            raise AudioValidationError(f"No audio stream found in {source.name}")

        container = data.get("format", {})
        return AudioMetadata(
            format_name=container.get("format_name", ""),
            codec=stream.get("codec_name", ""),
            duration_seconds=float(container.get("duration") or stream.get("duration") or 0.0),
            sample_rate=int(stream.get("sample_rate") or 0),
            channels=int(stream.get("channels") or 0),
            size_bytes=size,
        )

    def _validate_metadata(self, source: Path, metadata: AudioMetadata) -> None:
        expected = self._config.allowed_formats[source.suffix.lower()]
        detected = set(metadata.format_name.split(","))
        if not detected & expected:
            raise AudioValidationError(
                f"{source.name} has extension '{source.suffix}' but contains '{metadata.format_name}' data"
            )
        if metadata.duration_seconds <= 0:
            raise AudioValidationError(f"{source.name} has no measurable duration")
        if metadata.duration_seconds > self._config.max_duration_seconds:
            raise AudioValidationError(
                f"{source.name} is {metadata.duration_seconds:.0f}s long, "
                f"the limit is {self._config.max_duration_seconds:.0f}s"
            )

    def _normalize(self, source: Path) -> Path:
        target = self._work_dir / f"{source.stem}_{uuid.uuid4().hex[:8]}.wav"
        command = [
            self._ffmpeg,
            "-nostdin",
            "-y",
            "-v", "error",
            "-i", str(source),
            "-ac", str(self._config.target_channels),
            "-ar", str(self._config.target_sample_rate),
            "-c:a", "pcm_s16le",
            str(target),
        ]
        result = ffmpeg.run(command, self._config.convert_timeout_seconds)
        if result.returncode != 0 or not target.is_file():
            raise AudioProcessingError(f"ffmpeg failed to convert {source.name}: {result.stderr.strip()}")
        return target
