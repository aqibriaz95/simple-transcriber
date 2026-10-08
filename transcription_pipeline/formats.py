import json

from .pipeline import TranscriptionResult


def format_timestamp(seconds: float, separator: str = ",") -> str:
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def to_json(result: TranscriptionResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def to_srt(result: TranscriptionResult) -> str:
    blocks = [
        f"{index}\n{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n{segment.text}\n"
        for index, segment in enumerate(result.segments, start=1)
    ]
    return "\n".join(blocks)


def to_text(result: TranscriptionResult) -> str:
    return "\n".join(
        f"[{format_timestamp(segment.start, '.')} - {format_timestamp(segment.end, '.')}] {segment.text}"
        for segment in result.segments
    )


FORMATTERS = {"json": to_json, "srt": to_srt, "text": to_text}
