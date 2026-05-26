"""HTTP route that enqueues a transcription job.

Accepts an uploaded audio or video file, persists a `Job` row with
status `queued`, pushes the work onto the `transcribe` RQ queue, and
returns the job id so the client can poll `/jobs/{id}` for status and
`/jobs/{id}/result` for the final transcription.
"""

import uuid

from fastapi import APIRouter, UploadFile, HTTPException
from redis import Redis
from rq import Queue

from app.workers.transcribe_worker import transcribe_job
from app.db.session import SessionLocal
from app.db.models import Job
from app.config import settings


router = APIRouter()
_redis = Redis.from_url(settings.redis_url)
_queue = Queue("transcribe", connection=_redis)


@router.post("/transcribe", status_code=202)
async def transcribe(audio: UploadFile):
    """Accept an audio upload and enqueue the transcription work.

    The file is buffered to a temporary path on disk; the actual ASR
    work runs asynchronously in an RQ worker. The endpoint returns
    immediately with HTTP 202 Accepted and a job id.

    Args:
        audio: Uploaded file. Only the extensions listed below are
            accepted; the check is by filename, not content sniffing.

    Returns:
        A JSON object with `job_id` (UUID) and `status` (`queued`).
        Use `/jobs/{job_id}` to poll for progress and
        `/jobs/{job_id}/result` to fetch the transcription once done.

    Raises:
        HTTPException: 400 if the file extension is not supported.
    """
    if not audio.filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.mp4')):
        raise HTTPException(400, 'unsupported audio format')
    job_id = str(uuid.uuid4())
    tmp_path = f'/tmp/{job_id}_{audio.filename}'
    with open(tmp_path, 'wb') as buffer:
        buffer.write(await audio.read())

    db = SessionLocal()
    job = Job(
        id=job_id,
        audio_filename=audio.filename,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.close()

    _queue.enqueue(transcribe_job, job_id, file_path=tmp_path, job_timeout=600)
    return {"job_id": job_id, "status": "queued"}
