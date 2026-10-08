import math
import threading
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None

    def shifted(self, offset: float) -> "Segment":
        return replace(self, start=self.start + offset, end=self.end + offset)


@dataclass(frozen=True)
class Transcript:
    language: str | None
    segments: list[Segment]


def read_pcm_wav(audio_path: Path) -> np.ndarray:
    with wave.open(str(audio_path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            raise ValueError(f"{audio_path.name} is not 16-bit mono PCM")
        frames = audio.readframes(audio.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


class Transcriber(ABC):
    name: str = "transcriber"

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        raise NotImplementedError


class WhisperTranscriber(Transcriber):
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        self.name = f"faster-whisper/{model_size}"
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )
        return self._model

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        segments, info = self._get_model().transcribe(
            read_pcm_wav(audio_path),
            language=language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            word_timestamps=self._vad_filter,
        )
        results = [
            Segment(
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                text=segment.text.strip(),
                confidence=round(math.exp(segment.avg_logprob), 3),
            )
            for segment in segments
            if segment.text.strip()
        ]
        return Transcript(language=info.language, segments=results)


class MockTranscriber(Transcriber):
    name = "mock"

    DEFAULT_SCRIPT = (
        "Hello and welcome to the weekly product sync.",
        "Um, first item is the release date for version two.",
        "We agreed to move it to the fifteenth of next month.",
        "Please send feedback to support at example dot com.",
    )

    def __init__(self, script: tuple[str, ...] = DEFAULT_SCRIPT, language: str = "en"):
        self._script = script
        self._language = language

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        step = duration / len(self._script)
        segments = [
            Segment(
                start=round(index * step, 3),
                end=round((index + 1) * step, 3),
                text=line,
                confidence=0.95,
            )
            for index, line in enumerate(self._script)
        ]
        return Transcript(language=language or self._language, segments=segments)
