import pytest

from transcription_pipeline import AudioChunker, AudioLoader, ChunkingConfig
from transcription_pipeline.chunking import parse_silences, plan_cuts


def test_plan_cuts_returns_nothing_for_short_audio():
    assert plan_cuts(50.0, [], chunk_seconds=60, search_window_seconds=10) == []


def test_plan_cuts_prefers_latest_silence_inside_window():
    cuts = plan_cuts(100.0, [20.0, 52.0, 57.0, 95.0], chunk_seconds=60, search_window_seconds=10)
    assert cuts == [57.0]


def test_plan_cuts_falls_back_to_hard_cut_without_silence():
    cuts = plan_cuts(250.0, [10.0], chunk_seconds=100, search_window_seconds=5)
    assert cuts == [100.0, 200.0]


def test_plan_cuts_ignores_silence_outside_window():
    cuts = plan_cuts(150.0, [40.0], chunk_seconds=100, search_window_seconds=10)
    assert cuts == [100.0]


def test_parse_silences_pairs_start_and_end():
    log = (
        "[silencedetect @ 0x1] silence_start: 4.0\n"
        "[silencedetect @ 0x1] silence_end: 5.0 | silence_duration: 1.0\n"
        "[silencedetect @ 0x1] silence_start: -0.01\n"
        "[silencedetect @ 0x1] silence_end: 0.5 | silence_duration: 0.51\n"
    )
    assert parse_silences(log) == [4.5, 0.25]


def test_short_audio_is_not_split(wav_file):
    with AudioLoader() as loader:
        audio = loader.load(wav_file)
        chunks = AudioChunker(loader.work_dir).split(audio)
    assert len(chunks) == 1
    assert chunks[0].path == audio.normalized_path
    assert chunks[0].offset == 0.0


def test_long_audio_is_split_inside_pauses(long_wav_file):
    config = ChunkingConfig(chunk_seconds=10, search_window_seconds=3, min_silence_seconds=0.3)
    with AudioLoader() as loader:
        audio = loader.load(long_wav_file)
        chunks = AudioChunker(loader.work_dir, config).split(audio)
        assert all(chunk.path.is_file() for chunk in chunks)

    assert len(chunks) == 4
    for chunk in chunks[1:]:
        assert 4.0 <= chunk.offset % 5.0 <= 5.0
    assert sum(chunk.duration for chunk in chunks) == pytest.approx(audio.metadata.duration_seconds, abs=0.05)
    assert [chunk.offset for chunk in chunks] == sorted(chunk.offset for chunk in chunks)
