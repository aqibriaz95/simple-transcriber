from dataclasses import dataclass, field


def _default_formats() -> dict[str, frozenset[str]]:
    return {
        ".wav": frozenset({"wav"}),
        ".mp3": frozenset({"mp3"}),
    }


@dataclass(frozen=True)
class AudioConfig:
    allowed_formats: dict[str, frozenset[str]] = field(default_factory=_default_formats)
    max_file_size_bytes: int = 100 * 1024 * 1024
    max_duration_seconds: float = 2 * 60 * 60
    target_sample_rate: int = 16000
    target_channels: int = 1
    probe_timeout_seconds: float = 30.0
    convert_timeout_seconds: float = 600.0


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_seconds: float = 10 * 60
    search_window_seconds: float = 30.0
    silence_threshold_db: float = -35.0
    min_silence_seconds: float = 0.5
    detect_timeout_seconds: float = 600.0
    extract_timeout_seconds: float = 120.0
