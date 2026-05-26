from app.db.session import SessionLocal
from app.db.models import Transcript, Job
from datetime import datetime, timezone

_asr = None
def get_asr():
    global _asr
    if _asr is None:
        from app.core.asr import ASR
        _asr = ASR()
    return _asr

def transcribe_job(job_id: str, file_path: str):
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