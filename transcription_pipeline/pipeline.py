import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .audio import AudioLoader, AudioMetadata
from .chunking import AudioChunker
from .config import AudioConfig, ChunkingConfig
from .errors import TranscriptionError
from .transcriber import Segment, Transcriber

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    source: str
    model: str
    language: str | None
    audio: AudioMetadata
    segments: list[Segment]
    chunk_count: int
    processing_seconds: float

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "model": self.model,
            "language": self.language,
            "duration_seconds": round(self.audio.duration_seconds, 3),
            "chunk_count": self.chunk_count,
            "processing_seconds": round(self.processing_seconds, 3),
            "text": self.text,
            "segments": [{"id": index, **asdict(segment)} for index, segment in enumerate(self.segments)],
            "audio": asdict(self.audio),
        }


class TranscriptionPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        audio_config: AudioConfig | None = None,
        chunking_config: ChunkingConfig | None = None,
        language: str | None = None,
    ):
        self._transcriber = transcriber
        self._audio_config = audio_config or AudioConfig()
        self._chunking_config = chunking_config or ChunkingConfig()
        self._language = language

    def run(self, path: str | Path) -> TranscriptionResult:
        started = time.perf_counter()
        with AudioLoader(self._audio_config) as loader:
            audio = loader.load(path)
            chunks = AudioChunker(loader.work_dir, self._chunking_config).split(audio)
            language = self._language
            segments: list[Segment] = []
            for number, chunk in enumerate(chunks, start=1):
                logger.info("Transcribing chunk %d/%d at offset %.1fs", number, len(chunks), chunk.offset)
                try:
                    transcript = self._transcriber.transcribe(chunk.path, language=language)
                except Exception as exc:
                    raise TranscriptionError(
                        f"{self._transcriber.name} failed on chunk {number} at {chunk.offset:.1f}s: {exc}"
                    ) from exc
                language = language or transcript.language
                segments.extend(segment.shifted(chunk.offset) for segment in transcript.segments)
                if chunk.path != audio.normalized_path:
                    chunk.path.unlink(missing_ok=True)

        return TranscriptionResult(
            source=Path(path).name,
            model=self._transcriber.name,
            language=language,
            audio=audio.metadata,
            segments=segments,
            chunk_count=len(chunks),
            processing_seconds=time.perf_counter() - started,
        )
