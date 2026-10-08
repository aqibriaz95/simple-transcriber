import tempfile
from pathlib import Path
from typing import BinaryIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import AudioConfig, ChunkingConfig
from .errors import AudioProcessingError, AudioValidationError, TranscriptionError
from .pipeline import TranscriptionPipeline
from .transcriber import Transcriber, WhisperTranscriber

CHUNK_SIZE = 1024 * 1024


def _save_upload(source: BinaryIO, target: Path, max_bytes: int) -> int:
    written = 0
    with target.open("wb") as out:
        while chunk := source.read(CHUNK_SIZE):
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(status_code=413, detail=f"File exceeds the {max_bytes} byte limit")
            out.write(chunk)
    return written


def create_app(
    transcriber: Transcriber | None = None,
    audio_config: AudioConfig | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> FastAPI:
    transcriber = transcriber or WhisperTranscriber()
    audio_config = audio_config or AudioConfig()
    chunking_config = chunking_config or ChunkingConfig()
    app = FastAPI(title="Transcription Pipeline")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": transcriber.name}

    @app.post("/transcriptions")
    def transcribe(file: UploadFile = File(...), language: str | None = Form(None)) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in audio_config.allowed_formats:
            allowed = ", ".join(sorted(audio_config.allowed_formats))
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix}'. Allowed types: {allowed}",
            )

        pipeline = TranscriptionPipeline(transcriber, audio_config, chunking_config, language=language)
        with tempfile.TemporaryDirectory(prefix="upload_") as tmp:
            upload_path = Path(tmp) / f"upload{suffix}"
            _save_upload(file.file, upload_path, audio_config.max_file_size_bytes)
            try:
                result = pipeline.run(upload_path)
            except AudioValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except (AudioProcessingError, TranscriptionError) as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        payload = result.to_dict()
        payload["source"] = file.filename
        return payload

    return app


app = create_app()
