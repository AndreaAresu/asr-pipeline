"""RQ task that performs the actual transcription.

Picked up by an RQ worker subscribed to the `transcribe` queue. Drives
the `Job` row through its lifecycle (`queued` → `processing` → `done`
or `failed`) and, on success, stores the output in a `Transcript` row.
"""
import os
from datetime import datetime, timedelta, timezone

import structlog

from app.core.asr import ASRModel, TranscriptSegment
from app.core.chunking import chunk_segments
from app.core.embeddings import embed_batch
from app.core.logging import setup_logging, logger
from app.db.models import Chunk, Job, Transcript
from app.db.session import SessionLocal


setup_logging()

# A work-horse that dies without running its `finally` (OOM, SIGKILL, the
# macOS fork crash) leaves its job in `processing` forever. RQ's job_timeout
# is 600s, so any job still `processing` well past that can never complete.
STALE_PROCESSING_TIMEOUT = timedelta(seconds=900)

_asr: ASRModel | None = None


def reap_stale_jobs() -> int:
    """Fail jobs stuck in `processing` past `STALE_PROCESSING_TIMEOUT`.

    Called at worker startup to clean up rows orphaned by a previous
    worker that was killed mid-job. Returns the number of jobs reaped.
    """
    cutoff = datetime.now(timezone.utc) - STALE_PROCESSING_TIMEOUT
    db = SessionLocal()
    try:
        stale = db.query(Job).filter(
            Job.status == "processing",
            Job.started_at < cutoff,
        ).all()
        for job in stale:
            job.status = "failed"
            job.error_message = "worker terminated before completion"
            job.finished_at = datetime.now(timezone.utc)
        db.commit()
        return len(stale)
    finally:
        db.close()


def index_chunks(db, transcript_id: str, segments: list[TranscriptSegment]) -> int:
    """Chunk, embed and persist a transcript's segments for search.

    Splits the Whisper segments into overlapping time windows, embeds all
    of them in a single batch (much faster than one call per chunk), and
    adds a `Chunk` row per window. The rows are added to `db` but not
    committed — the caller commits them together with the job's final
    state, so a transcript is never marked `done` with a half-written
    index.

    Args:
        db: Open session; rows are added to it, not committed.
        transcript_id: Owning transcript, already flushed so its id exists.
        segments: Ordered Whisper segments to index.

    Returns:
        Number of chunks indexed.
    """
    chunks = list(chunk_segments(segments))
    if not chunks:
        return 0

    embeddings = embed_batch([c["text"] for c in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        db.add(
            Chunk(
                transcript_id=transcript_id,
                chunk_index=chunk["chunk_index"],
                start_sec=chunk["start_sec"],
                end_sec=chunk["end_sec"],
                text=chunk["text"],
                embedding=embedding,
            )
        )
    return len(chunks)


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


def transcribe_job(job_id: str, file_path: str, request_id: str | None = None) -> None:
    """Transcribe a single audio file and persist the result.

    Marks the job as `processing` on entry, runs the ASR model, then on
    success persists a `Transcript` row and flips the job to `done`. On
    any Python exception, flips the job to `failed` with the exception
    message and re-raises so RQ records the failure in its own registry.

    Args:
        job_id: Primary key of the `Job` row to update.
        file_path: Local path to the audio file to transcribe.
        request_id: Correlation id propagated from the API request that
            enqueued this job, so worker logs can be tied back to it.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job_id=job_id)
    if request_id:
        structlog.contextvars.bind_contextvars(request_id=request_id)

    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("transcribe.started")

        result = get_asr().transcribe(file_path)

        transcript = Transcript(
            job_id=job_id,
            full_text=result.full_text,
            language=result.language,
            word_timestamps=[s.model_dump() for s in result.segments],
        )
        db.add(transcript)
        # Flush so transcript.id is assigned before the chunks reference it.
        db.flush()

        indexed = index_chunks(db, transcript.id, result.segments)

        job.status = "done"
        job.duration = result.duration
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            "transcribe.completed",
            duration=result.duration,
            language=result.language,
            transcript_id=transcript.id,
            chunks_indexed=indexed,
        )
    except Exception as e:
        # Discard whatever partial state the failed unit of work left in the
        # session (a half-written transcript, an aborted transaction) before
        # recording the failure, otherwise this commit fails too.
        db.rollback()
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "failed"
            job.error_message = str(e)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
        logger.error("transcribe.failed", error=str(e))
        raise
    finally:
        db.close()
        if os.path.exists(file_path):
            os.remove(file_path)
