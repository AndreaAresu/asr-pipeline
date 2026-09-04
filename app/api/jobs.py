"""HTTP routes for inspecting transcription jobs.

Exposes status and result endpoints used by clients that previously
called `/transcribe` and want to know whether the job has completed.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import get_api_key
from app.db.models import ApiKey, Job, Transcript
from app.db.session import SessionLocal

router = APIRouter()


@router.get("/jobs")
def list_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum jobs to return."),
    api_key: ApiKey = Depends(get_api_key),
):
    """List the caller's most recent jobs, newest first.

    Scoped to the authenticated key: a caller sees the jobs it submitted
    and nothing else. Exists so a client can rebuild its view of work in
    flight without having remembered every job id, the web UI relies on
    it after a page reload.

    Args:
        limit: Maximum number of jobs to return.
        api_key: Authenticated caller, resolved from the `X-API-Key`
            header.

    Returns:
        A list of jobs with their status, filename, duration, timestamps,
        and `transcript_id` once one exists (null until the job is done).

    Raises:
        HTTPException: 401 if the `X-API-Key` header is missing or
            invalid.
    """
    db = SessionLocal()
    try:
        jobs = (
            db.query(Job)
            .filter(Job.api_key_hash == api_key.key_hash)
            .order_by(Job.created_at.desc())
            .limit(limit)
            .all()
        )
        # Resolve transcript ids in one query rather than one per job.
        job_ids = [job.id for job in jobs]
        transcripts = (
            db.query(Transcript.job_id, Transcript.id)
            .filter(Transcript.job_id.in_(job_ids))
            .all()
            if job_ids
            else []
        )
        by_job = dict(transcripts)

        return [
            {
                "id": job.id,
                "status": job.status,
                "audio_filename": job.audio_filename,
                "duration": job.duration,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "transcript_id": by_job.get(job.id),
            }
            for job in jobs
        ]
    finally:
        db.close()


def _owned_job(db, job_id: str, api_key: ApiKey) -> Job:
    """Fetch a job, but only for the key that submitted it.

    A job that exists under another key is reported as **404, not 403**:
    telling an unauthorized caller "this exists but is not yours" turns
    the endpoint into an oracle for probing which ids are real. From
    outside, someone else's job and a nonexistent one are indistinguishable.

    Raises:
        HTTPException: 404 if the job does not exist or belongs to
            another key.
    """
    job = db.get(Job, job_id)
    if job is None or job.api_key_hash != api_key.key_hash:
        raise HTTPException(404, "job not found")
    return job


@router.get("/jobs/{job_id}")
def get_job(job_id: str, api_key: ApiKey = Depends(get_api_key)):
    """Return the current status of a transcription job.

    Args:
        job_id: UUID assigned when the job was enqueued by `/transcribe`.
        api_key: Authenticated caller, resolved from the `X-API-Key`
            header. Must be the key that submitted the job.

    Returns:
        A JSON object with `id`, `status` (`queued` | `processing` |
        `done` | `failed`), `error_message` (populated when
        `status == 'failed'`), and `duration` in seconds (populated when
        `status == 'done'`).

    Raises:
        HTTPException: 401 if the `X-API-Key` header is missing or
            invalid; 404 if no such job exists for this key.
    """
    db = SessionLocal()
    try:
        job = _owned_job(db, job_id, api_key)
        return {
            "id": job.id,
            "status": job.status,
            "error_message": job.error_message,
            "duration": job.duration,
        }
    finally:
        db.close()


@router.get("/jobs/{job_id}/result")
def get_result(job_id: str, api_key: ApiKey = Depends(get_api_key)):
    """Return the transcription output for a completed job.

    Args:
        job_id: UUID of a job whose `status == 'done'`.
        api_key: Authenticated caller, resolved from the `X-API-Key`
            header. Must be the key that submitted the job.

    Returns:
        A JSON object with `transcript_id` (the handle `/search` results
        and `/summarize/{transcript_id}` are keyed by, distinct from the
        job id), `full_text`, detected `language`, and `segments`
        (per-segment text with word-level alignment, as stored in
        `Transcript.word_timestamps`).

    Raises:
        HTTPException: 401 if the `X-API-Key` header is missing or
            invalid; 404 if no such job exists for this key; 400 if the
            job exists but has not finished successfully yet.
    """
    db = SessionLocal()
    try:
        job = _owned_job(db, job_id, api_key)
        if job.status != "done":
            raise HTTPException(400, f"job not completed: status={job.status}")
        t = db.query(Transcript).filter_by(job_id=job_id).first()
        return {
            "transcript_id": t.id,
            "full_text": t.full_text,
            "language": t.language,
            "segments": t.word_timestamps,
        }
    finally:
        db.close()
