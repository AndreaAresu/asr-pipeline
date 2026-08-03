"""HTTP routes for inspecting transcription jobs.

Exposes status and result endpoints used by clients that previously
called `/transcribe` and want to know whether the job has completed.
"""

from fastapi import APIRouter, HTTPException

from app.db.session import SessionLocal
from app.db.models import Job, Transcript


router = APIRouter()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Return the current status of a transcription job.

    Args:
        job_id: UUID assigned when the job was enqueued by `/transcribe`.

    Returns:
        A JSON object with `id`, `status` (`queued` | `processing` |
        `done` | `failed`), `error_message` (populated when
        `status == 'failed'`), and `duration` in seconds (populated when
        `status == 'done'`).

    Raises:
        HTTPException: 404 if no job with the given id exists.
    """
    db = SessionLocal()
    job = db.get(Job, job_id)
    db.close()

    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "id": job.id,
        "status": job.status,
        "error_message": job.error_message,
        "duration": job.duration,
    }


@router.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    """Return the transcription output for a completed job.

    Args:
        job_id: UUID of a job whose `status == 'done'`.

    Returns:
        A JSON object with `transcript_id` (the handle `/search` results
        and `/summarize/{transcript_id}` are keyed by — distinct from the
        job id), `full_text`, detected `language`, and `segments`
        (per-segment text with word-level alignment, as stored in
        `Transcript.word_timestamps`).

    Raises:
        HTTPException: 404 if the job does not exist; 400 if the job
            exists but has not finished successfully yet.
    """
    db = SessionLocal()
    job = db.get(Job, job_id)
    if job is None:
        db.close()
        raise HTTPException(404, "job not found")
    if job.status != "done":
        db.close()
        raise HTTPException(400, f"job not completed: status={job.status}")
    t = db.query(Transcript).filter_by(job_id=job_id).first()
    db.close()
    return {
        "transcript_id": t.id,
        "full_text": t.full_text,
        "language": t.language,
        "segments": t.word_timestamps,
    }
