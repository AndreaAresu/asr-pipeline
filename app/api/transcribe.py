"""HTTP route that enqueues a transcription job.

Accepts an uploaded audio or video file, persists a `Job` row with
status `queued`, pushes the work onto the `transcribe` RQ queue, and
returns the job id so the client can poll `/jobs/{id}` for status and
`/jobs/{id}/result` for the final transcription.
"""

import os
import uuid

from fastapi import APIRouter, UploadFile, HTTPException, Depends
from redis import Redis
from rq import Queue
from starlette.concurrency import run_in_threadpool

from app.core.auth import get_api_key
from app.core.logging import logger
from app.core.rate_limit import probe_duration, enforce_quota
from app.workers.transcribe_worker import transcribe_job
from app.db.session import SessionLocal
from app.db.models import ApiKey, Job
from app.config import settings


router = APIRouter()
_redis = Redis.from_url(settings.redis_url)
_queue = Queue("transcribe", connection=_redis)


@router.post("/transcribe", status_code=202)
async def transcribe(audio: UploadFile, api_key: ApiKey = Depends(get_api_key)):
    """Accept an audio upload and enqueue the transcription work.

    The file is buffered to a temporary path on disk; the actual ASR
    work runs asynchronously in an RQ worker. The endpoint returns
    immediately with HTTP 202 Accepted and a job id.

    Args:
        audio: Uploaded file. Only the extensions listed below are
            accepted; the check is by filename, not content sniffing.
        api_key: Authenticated caller, resolved from the `X-API-Key`
            header. Its `key_hash` is recorded on the job for rate
            limiting.

    Returns:
        A JSON object with `job_id` (UUID) and `status` (`queued`).
        Use `/jobs/{job_id}` to poll for progress and
        `/jobs/{job_id}/result` to fetch the transcription once done.

    Raises:
        HTTPException: 401 if the `X-API-Key` header is missing or
            invalid; 400 if the file extension is not supported or its
            duration cannot be read; 429 if the caller's rolling 24h
            audio quota would be exceeded.
    """
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.mp4')):
        raise HTTPException(400, 'unsupported audio format')
    job_id = str(uuid.uuid4())
    tmp_path = f'/tmp/{job_id}_{audio.filename}'
    with open(tmp_path, 'wb') as buffer:
        buffer.write(await audio.read())

    # Measure the upload and check the caller's quota before accepting the
    # job; clean up the temp file if either step rejects the request.
    try:
        duration = await run_in_threadpool(probe_duration, tmp_path)

        db = SessionLocal()
        try:
            enforce_quota(db, api_key, duration)
            job = Job(
                id=job_id,
                api_key_hash=api_key.key_hash,
                audio_filename=audio.filename,
                duration=duration,
                status="queued",
            )
            db.add(job)
            db.commit()
        finally:
            db.close()
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    _queue.enqueue(transcribe_job, job_id, file_path=tmp_path, job_timeout=600)
    logger.info(
        "transcribe.enqueued",
        job_id=job_id,
        api_key_hash=api_key.key_hash,
        filename=audio.filename,
        duration=duration,
    )
    return {"job_id": job_id, "status": "queued"}
