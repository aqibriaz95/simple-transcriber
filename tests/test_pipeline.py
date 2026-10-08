from pathlib import Path

import pytest

from transcription_pipeline import (
    ChunkingConfig,
    MockTranscriber,
    Segment,
    Transcriber,
    Transcript,
    TranscriptionError,
    TranscriptionPipeline,
)
from transcription_pipeline.formats import format_timestamp, to_srt, to_text


class RecordingTranscriber(Transcriber):
    name = "recording"

    def __init__(self):
        self.languages: list[str | None] = []

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        self.languages.append(language)
        return Transcript(language="de", segments=[Segment(0.5, 1.5, f"chunk {len(self.languages)}")])


class FailingTranscriber(Transcriber):
    name = "failing"

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        raise RuntimeError("model crashed")


def test_mock_pipeline_returns_timestamped_segments(mp3_file):
    result = TranscriptionPipeline(MockTranscriber()).run(mp3_file)
    data = result.to_dict()

    assert data["model"] == "mock"
    assert data["language"] == "en"
    assert data["chunk_count"] == 1
    assert len(data["segments"]) == 4
    assert [segment["id"] for segment in data["segments"]] == [0, 1, 2, 3]
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][-1]["end"] == pytest.approx(1.0, abs=0.1)
    assert data["text"].startswith("Hello and welcome")


def test_segments_from_later_chunks_are_offset(long_wav_file):
    transcriber = RecordingTranscriber()
    config = ChunkingConfig(chunk_seconds=10, search_window_seconds=3, min_silence_seconds=0.3)
    result = TranscriptionPipeline(transcriber, chunking_config=config).run(long_wav_file)

    assert result.chunk_count == 4
    starts = [segment.start for segment in result.segments]
    assert starts[0] == 0.5
    assert starts == sorted(starts)
    assert all(later - earlier > 5 for earlier, later in zip(starts, starts[1:]))


def test_language_detected_in_first_chunk_is_reused(long_wav_file):
    transcriber = RecordingTranscriber()
    config = ChunkingConfig(chunk_seconds=10, search_window_seconds=3, min_silence_seconds=0.3)
    result = TranscriptionPipeline(transcriber, chunking_config=config).run(long_wav_file)

    assert transcriber.languages == [None, "de", "de", "de"]
    assert result.language == "de"


def test_explicit_language_is_passed_to_every_chunk(wav_file):
    transcriber = RecordingTranscriber()
    TranscriptionPipeline(transcriber, language="en").run(wav_file)
    assert transcriber.languages == ["en"]


def test_transcriber_failure_is_wrapped(wav_file):
    with pytest.raises(TranscriptionError, match="failing failed on chunk 1"):
        TranscriptionPipeline(FailingTranscriber()).run(wav_file)


def test_format_timestamp():
    assert format_timestamp(3725.042) == "01:02:05,042"
    assert format_timestamp(0.5, ".") == "00:00:00.500"


def test_srt_and_text_output(wav_file):
    result = TranscriptionPipeline(MockTranscriber()).run(wav_file)
    srt = to_srt(result)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,250\nHello and welcome")
    assert to_text(result).splitlines()[1].startswith("[00:00:00.250 - 00:00:00.500] Um,")
