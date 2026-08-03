"""HTTP route that enqueues a transcription job.

Accepts an uploaded audio or video file, persists a `Job` row with
status `queued`, pushes the work onto the `transcribe` RQ queue, and
returns the job id so the client can poll `/jobs/{id}` for status and
`/jobs/{id}/result` for the final transcription.
"""

import os
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from redis import Redis
from rq import Queue
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.core.auth import get_api_key
from app.core.logging import logger
from app.core.rate_limit import enforce_quota, probe_duration
from app.db.models import ApiKey, Job
from app.db.session import SessionLocal
from app.workers.transcribe_worker import transcribe_job

router = APIRouter()
_redis = Redis.from_url(settings.redis_url)
_queue = Queue("transcribe", connection=_redis)

# Whisper on CPU runs at roughly 0.5-1x real time, so transcription time is
# proportional to the recording, not constant. A fixed timeout is therefore
# wrong at both ends: generous for a 30s clip and fatal for a 1h interview,
# where RQ would kill the work-horse mid-job and leave the row orphaned in
# `processing` until the reaper catches it.
TRANSCRIBE_TIMEOUT_FACTOR = 5
MIN_TRANSCRIBE_TIMEOUT = 600


def job_timeout_for(duration_sec: float) -> int:
    """Return the RQ timeout to allow for `duration_sec` of audio.

    Five times the audio duration, floored at ten minutes: comfortably
    above the ~1-2x observed on CPU, while still bounding a job that has
    genuinely hung rather than letting it occupy the worker forever.
    """
    return max(MIN_TRANSCRIBE_TIMEOUT, int(duration_sec * TRANSCRIBE_TIMEOUT_FACTOR))


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
        logger.info(
            "transcribe.rejected",
            reason="unsupported_format",
            status_code=400,
            filename=audio.filename,
        )
        raise HTTPException(400, 'unsupported audio format')
    job_id = str(uuid.uuid4())
    # The worker is a different process — and under compose a different
    # container — so the upload has to land somewhere both of them can see.
    # temp_audio_dir is that shared spool; the worker deletes the file when
    # the job ends.
    os.makedirs(settings.temp_audio_dir, exist_ok=True)
    tmp_path = os.path.join(settings.temp_audio_dir, f"{job_id}_{os.path.basename(audio.filename)}")
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
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if isinstance(e, HTTPException):
            logger.info(
                "transcribe.rejected",
                reason=e.detail,
                status_code=e.status_code,
                filename=audio.filename,
                api_key_hash=api_key.key_hash,
            )
        raise

    request_id = structlog.contextvars.get_contextvars().get("request_id")
    timeout = job_timeout_for(duration)
    _queue.enqueue(
        transcribe_job, job_id,
        file_path=tmp_path, request_id=request_id, job_timeout=timeout,
    )
    logger.info(
        "transcribe.enqueued",
        job_id=job_id,
        api_key_hash=api_key.key_hash,
        filename=audio.filename,
        duration=duration,
        job_timeout=timeout,
    )
    return {"job_id": job_id, "status": "queued"}
