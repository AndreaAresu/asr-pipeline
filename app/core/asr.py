"""Synchronous wrapper around faster-whisper for audio transcription.

Exposes the `ASRModel` class and the Pydantic models that describe the
transcription payload returned to API clients.
"""

from faster_whisper import WhisperModel
from pydantic import BaseModel, Field


class TranscriptWord(BaseModel):
    """A single recognized word with timing and confidence."""

    word: str = Field(description="The recognized word as emitted by Whisper (may include leading whitespace).")
    start: float = Field(description="Start time of the word, in seconds from the beginning of the audio.")
    end: float = Field(description="End time of the word, in seconds from the beginning of the audio.")
    probability: float = Field(description="Model confidence for the word, in the range [0.0, 1.0].")


class TranscriptSegment(BaseModel):
    """A contiguous segment of speech with per-word alignment."""

    text: str = Field(description="The transcribed text of the segment.")
    start: float = Field(description="Start time of the segment, in seconds from the beginning of the audio.")
    end: float = Field(description="End time of the segment, in seconds from the beginning of the audio.")
    words: list[TranscriptWord] = Field(description="Word-level breakdown of the segment with timing and confidence.")


class TranscriptResult(BaseModel):
    """Full transcription output for a single audio input."""

    language: str = Field(description="Detected language as an ISO 639-1 code (e.g. 'en'), auto-detected by Whisper.")
    duration: float = Field(description="Duration of the source audio, in seconds.")
    segments: list[TranscriptSegment] = Field(description="Ordered list of transcribed segments.")
    full_text: str = Field(description="Concatenation of all segment texts, joined by single spaces and trimmed.")


class ASRModel:
    """Synchronous wrapper around `faster_whisper.WhisperModel`.

    Loading the underlying model is expensive (hundreds of MB of memory and
    several seconds of startup), so an instance of this class should be
    created once at process startup and reused for all requests — typically
    as a FastAPI dependency or a module-level singleton.
    """

    def __init__(self, model_size: str = "small.en"):
        """Load the Whisper model into memory.

        Args:
            model_size: Identifier of the Whisper model to load
                (e.g. "small.en", "medium", "large-v3"). See the
                faster-whisper README for the full list of supported sizes.

        The model is loaded on CPU with int8 quantization to keep the memory
        footprint low at a small accuracy cost; revisit these defaults when
        running on GPU or when higher accuracy is required.
        """
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio_path: str) -> TranscriptResult:
        """Transcribe an audio file and return a structured result.

        Args:
            audio_path: Path to an audio or video file. Any container or
                codec supported by ffmpeg is accepted (mp3, wav, m4a, mp4, ...).

        Returns:
            A `TranscriptResult` with the detected language, source duration,
            per-segment breakdown with word-level timestamps, and a
            convenience `full_text` field.

        Raises:
            FileNotFoundError: If `audio_path` does not exist.
            RuntimeError: If ffmpeg fails to decode the input file.

        Note:
            This method is blocking and CPU-bound. When called from an async
            context (e.g. a FastAPI endpoint), dispatch it through
            `starlette.concurrency.run_in_threadpool` or an equivalent
            executor to avoid blocking the event loop.
        """
        segments_iter, info = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
        )

        segments = []
        full_text_parts = []
        for s in segments_iter:
            # faster-whisper may emit segments with words=None after VAD filtering
            words = [
                TranscriptWord(
                    word=w.word, start=w.start, end=w.end,
                    probability=w.probability,
                )
                for w in (s.words or [])
            ]
            segments.append(TranscriptSegment(text=s.text, start=s.start, end=s.end, words=words))
            full_text_parts.append(s.text)
        return TranscriptResult(
            language=info.language,
            duration=info.duration,
            segments=segments,
            full_text=" ".join(full_text_parts).strip(),
        )
