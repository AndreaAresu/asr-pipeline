from faster_whisper import WhisperModel
from pydantic import BaseModel


class TranscriptWord(BaseModel):
    word: str
    start: float
    end: float
    probability: float

class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    words: list[TranscriptWord]

class TranscriptResult(BaseModel):
    language: str
    duration: float
    segment: list[TranscriptSegment]
    full_text: str

class ASRModel():
    def __init__(self, model_size: str = "small.en"):
        self.model = WhisperModel(
            model_size, 
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio_path: str) -> TranscriptResult:
        segments_iter, info = self.model.transcribe(
            audio_path, 
            word_timestamps=True, 
            vad_filter=True,
        )

        segments = []
        full_text_parts = []
        for s in segments_iter:
            words = [TranscriptWord(
                word=w.word, start=w.start, end=w.end,
                probability=w.probability
            ) for w in (s.words or [])]
            segments.append(TranscriptSegment(text=s.text, start=s.start, end=s.end, words=words))
            full_text_parts.append(s.text)
        return TranscriptResult(
            language=info.language, 
            duration=info.duration,
            segment=segments,
            full_text=" ".join(full_text_parts).strip()
        )



