"""RQ task that performs the actual transcription.

Picked up by an RQ worker subscribed to the `transcribe` queue. Drives
the `Job` row through its lifecycle (`queued` → `processing` → `done`
or `failed`) and, on success, stores the output in a `Transcript` row.
"""
import os 
from datetime import datetime, timezone

from app.core.asr import ASRModel
from app.db.models import Job, Transcript
from app.db.session import SessionLocal


_asr: ASRModel | None = None


def get_asr() -> ASRModel:
    """Return a lazily-instantiated process-wide `ASRModel`.

    The model is loaded on first call and cached for the lifetime of the
    worker process, so subsequent jobs in the same process do not pay
    the multi-hundred-MB load cost again. Each forked work-horse (the
    default RQ execution model) loads its own copy.
    """
    global _asr
    if _asr is None:
        _asr = ASRModel()
    return _asr


def transcribe_job(job_id: str, file_path: str) -> None:
    """Transcribe a single audio file and persist the result.

    Marks the job as `processing` on entry, runs the ASR model, then on
    success persists a `Transcript` row and flips the job to `done`. On
    any Python exception, flips the job to `failed` with the exception
    message and re-raises so RQ records the failure in its own registry.

    Args:
        job_id: Primary key of the `Job` row to update.
        file_path: Local path to the audio file to transcribe.
    """
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        result = get_asr().transcribe(file_path)

        transcript = Transcript(
            job_id=job_id,
            full_text=result.full_text,
            language=result.language,
            word_timestamps=[s.model_dump() for s in result.segments],
        )
        db.add(transcript)
        job.status = "done"
        job.duration = result.duration
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        raise
    finally:
        db.close()
        if os.path.exists(file_path):
            os.remove(file_path)
