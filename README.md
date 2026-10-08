# Transcription pipeline: design decisions

The pipeline takes a WAV or MP3 file and works in four steps. It checks and converts the audio, splits long files, transcribes each piece with faster-whisper, and joins the results into timestamped segments. It runs from the command line or as a FastAPI service, and both use the same code.

```bash
pip install -r requirements.txt
python -m transcription_pipeline.cli samples/jfk.wav --format text
uvicorn transcription_pipeline.api:app
```

You also need ffmpeg installed. Run `python -m transcription_pipeline.cli --help` to see the options.

## What the output looks like

I started by deciding what the next system needs from a transcript. It needs the full text, the segments with start and end times so it can link text back to the audio, a confidence score so it can flag weak parts for a person to check, and the language. The model name is in there too, so if we upgrade the model later we know which transcripts came from which one.

```json
{
  "model": "faster-whisper/base",
  "language": "en",
  "duration_seconds": 11.0,
  "text": "And so my fellow Americans, ask not what your country can do for you, ...",
  "segments": [
    {"id": 0, "start": 0.0, "end": 10.36, "text": "And so my fellow Americans, ...", "confidence": 0.757}
  ]
}
```

The same result can come out as SRT subtitles or plain text with `--format`.

## One audio format inside the pipeline

ffmpeg converts every upload to 16 kHz mono WAV before anything else happens, because that's what Whisper wants anyway. After that point nothing downstream has to care whether the user sent an MP3 or a stereo WAV. Adding FLAC or M4A would be one line in `config.py`.

I check the file with ffprobe before converting it. The extension is whatever the client says it is, so I don't trust it. A text file renamed to `.wav` gets rejected with a proper error instead of crashing somewhere inside the model.

## faster-whisper, run locally

I used faster-whisper rather than a hosted API like Deepgram or Google. It's free to run, it's fast enough on a CPU with 8-bit weights, and the audio never leaves the machine, which matters if the recordings are private.

It's wrapped in a `Transcriber` interface with a single `transcribe` method. The tests use a mock that implements the same interface, so they never download a model. If we wanted a hosted API later, that's one new class.

I went with the base model as a middle ground. Tiny makes more mistakes, and small or medium are a lot slower on a CPU. Honestly, that choice should come from measuring accuracy on real audio, which I haven't done yet.

Voice activity detection is turned on, so silent stretches get skipped. That saves time, and silence is where Whisper is most likely to make up text. On a 142-second test file that was mostly silence, it cut out just over 2 minutes and the model only decoded about 20 seconds.

That test also showed a bug. VAD glues the speech parts together before decoding, and a segment that started right after a glued gap got mapped back to the wrong side of it. One segment claimed to start at 1:10, during the silence, when the speech really started at 2:11. Turning on word-level timestamps fixed it, because faster-whisper then maps each word back to the original audio separately. It added about 3% to the run time.

## Long files get split at pauses

This was the most interesting part.

A 2-hour file loaded into memory is around 460 MB of samples. A 10-minute chunk is about 38 MB, and if something fails you only lose that chunk. So anything over 10 minutes gets split.

The catch is where to cut. If you cut at exactly 10:00 you'll probably land in the middle of a word, and the model gets it wrong on both sides. So I run ffmpeg's `silencedetect` filter first to find the pauses, and cut at the last pause in the 30 seconds before each 10-minute mark. If there isn't one, it falls back to a hard cut.

I thought about overlapping the chunks instead, but then the same words show up twice and you need logic to remove the duplicates. Cutting in a pause avoids that most of the time.

Two smaller things in the same area. Every chunk's timestamps start at zero, so the pipeline adds the chunk's start time to each segment. And the language detected on the first chunk gets reused for the rest, so a chunk of background music can't suddenly switch the transcript to another language.

I tested this on the JFK clip repeated three times with short pauses, using 15-second chunks. It cut at 11.8s and 24.3s, both inside the pauses, and the later timestamps lined up with the original file.

## ffmpeg does all the decoding

The first time I ran real audio through the model, it crashed. faster-whisper decodes files using PyAV, and the latest PyAV removed an argument that faster-whisper still passes. I could have pinned an older PyAV, but I was already converting everything to WAV with ffmpeg. So now I read that WAV myself and hand the samples to the model as an array, and ffmpeg is the only decoder in the system.

## Bad input vs. our own failures

There are separate error types for "the user sent something bad" and "something broke on our side". The CLI exits with 2 or 1, and the API returns a 4xx or a 500. In a production version this is also what would decide whether a job gets retried. There's no point retrying a corrupt file.

The API handler is a plain `def`, not `async def`. It's calling ffmpeg and running a model, both of which block. FastAPI runs plain functions in a thread pool, so one slow request doesn't freeze the others.

## Tests that don't need real recordings

The 31 tests generate their own audio with ffmpeg's tone generator, including a 32-second file with regular gaps for the chunking tests. With the mock transcriber, they run in about 10 seconds with no model download, so they're cheap to run on every change.

## What I'd change next

- The API transcribes inside the request. Fine for short clips, but an hour of audio would time out. The fix is to return a job id straight away and let workers process a queue.
- If a long file fails on its last chunk, the whole thing starts over. Saving each chunk's result as it finishes would let a retry pick up where it stopped.
- I haven't measured accuracy. I'd want a word error rate on real user audio before picking a model size for production.
- The hard-cut fallback can still split a word when there's no pause for a long stretch. A small overlap at those cuts only would fix it.
- The model loads on the first request, so that one is slow. It should load at startup.
- No speaker labels, and no clean-up of filler words or personal data like phone numbers yet.

My Part 2 answers cover the production side: queues, storage, retries and the public API.
