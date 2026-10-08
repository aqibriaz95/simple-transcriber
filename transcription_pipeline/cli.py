import argparse
import logging
import sys
from pathlib import Path

from .config import ChunkingConfig
from .errors import AudioValidationError, PipelineError
from .formats import FORMATTERS
from .pipeline import TranscriptionPipeline
from .transcriber import MockTranscriber, Transcriber, WhisperTranscriber


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe a WAV or MP3 file into timestamped text.",
    )
    parser.add_argument("audio_path", help="Path to a WAV or MP3 file")
    parser.add_argument("--model", default="base", help="faster-whisper model size, e.g. tiny, base, small")
    parser.add_argument("--language", help="Language code such as 'en'. Detected when omitted")
    parser.add_argument("--mock", action="store_true", help="Use the mock transcriber instead of Whisper")
    parser.add_argument("--format", choices=sorted(FORMATTERS), default="json", help="Output format")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=ChunkingConfig.chunk_seconds,
        help="Split files longer than this into chunks. Default 600",
    )
    parser.add_argument("--output", type=Path, help="Write the result to this file instead of stdout")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def build_transcriber(args: argparse.Namespace) -> Transcriber:
    if args.mock:
        return MockTranscriber()
    return WhisperTranscriber(model_size=args.model)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    pipeline = TranscriptionPipeline(
        transcriber=build_transcriber(args),
        chunking_config=ChunkingConfig(chunk_seconds=args.chunk_seconds),
        language=args.language,
    )
    try:
        result = pipeline.run(args.audio_path)
    except AudioValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rendered = FORMATTERS[args.format](result)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
