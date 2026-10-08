class PipelineError(Exception):
    pass


class AudioValidationError(PipelineError):
    pass


class AudioProcessingError(PipelineError):
    pass


class TranscriptionError(PipelineError):
    pass
