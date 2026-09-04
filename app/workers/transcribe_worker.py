"""RQ task that performs the actual transcription.

Picked up by an RQ worker subscribed to the `transcribe` queue. Drives
the `Job` row through its lifecycle (`queued` -> `processing` -> `done`
or `failed`) and, on success, stores the output in a `Transcript` row.
"""
import os
from datetime import UTC, datetime, timedelta

import structlog

from app.core.asr import ASRModel, TranscriptSegment
from app.core.chunking import chunk_segments
from app.core.embeddings import embed_batch
from app.core.logging import logger, setup_logging
from app.db.models import Chunk, Job, Transcript
from app.db.session import SessionLocal

setup_logging()

# A work-horse that dies without running its `finally` (OOM, SIGKILL, the
# macOS fork crash) leaves its job in `processing` forever.
#
# How long "forever" starts depends on the recording: the enqueued timeout
# scales with audio duration (see `job_timeout_for`), so a 4-hour interview
# is legitimately `processing` for hours. A fixed cutoff would reap it
# mid-transcription and report a failure that never happened. The budget
# below mirrors the enqueue-side timeout, plus a grace margin.
STALE_TIMEOUT_FACTOR = 5
MIN_STALE_TIMEOUT = timedelta(seconds=900)
STALE_GRACE = timedelta(seconds=300)


def stale_cutoff_for(duration_sec: float | None) -> timedelta:
    """Return how long a job of this length may sit in `processing`."""
    budget = timedelta(seconds=(duration_sec or 0) * STALE_TIMEOUT_FACTOR)
    return max(MIN_STALE_TIMEOUT, budget) + STALE_GRACE


def killing_signal(ret_val: object) -> int | None:
    """Extract the killing signal from a wait status, if there is one.

    RQ passes `None` on the paths where it noticed the death without
    reaping a wait status, so the value cannot be assumed numeric, doing
    the arithmetic unguarded raises, and the failure being reported is
    lost along with it.
    """
    if not isinstance(ret_val, int):
        return None
    return ret_val & 0xFF


def termination_reason(signal_number: int | None) -> str:
    """Phrase why a job died, naming the signal when it is known."""
    killed_by = f"by signal {signal_number} " if signal_number is not None else ""
    return (
        f"worker process terminated {killed_by}before it could report; "
        "on a memory-constrained host this is usually the OOM killer"
    )


def handle_work_horse_killed(rq_job, retpid: int | None, ret_val: int | None, rusage) -> None:
    """Record a job whose work-horse died before it could report anything.

    When the work-horse is killed outright (the OOM killer, SIGKILL, a
    segfault in a native library) no Python inside it runs. The task's own
    `except` block never fires, so the row keeps claiming `processing` and
    a polling client waits forever on a job that is already dead.

    This must be a *worker*-level hook, not the job's `on_failure`
    callback: RQ runs failure callbacks inside the work-horse, which is
    precisely the process that just died. `Worker(work_horse_killed_handler=...)`
    runs here, in the parent, which survives.

    Deliberately defensive, an exception raised here would be reported
    instead of the failure it is trying to record.

    Args:
        rq_job: The RQ job whose work-horse died; its first argument is
            our `Job.id`.
        retpid: PID of the dead work-horse, or None when RQ noticed the
            death by a route that never reaped a wait status.
        ret_val: Its wait status, or None as above. When present, the low
            byte carries the killing signal.
        rusage: Resource usage of the dead process (unused).
    """
    try:
        job_id = rq_job.args[0]
        signal_number = killing_signal(ret_val)
        reason = termination_reason(signal_number)

        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
            # Only claim the job if it never reached a terminal state, the
            # task's own handler is more specific, and wins when it ran.
            if job is not None and job.status not in ("done", "failed"):
                job.status = "failed"
                job.error_message = reason
                job.finished_at = datetime.now(UTC)
                db.commit()
                logger.error(
                    "transcribe.abandoned", job_id=job_id, signal=signal_number, pid=retpid
                )
        finally:
            db.close()
    except Exception as e:  # pragma: no cover - must never mask the real failure
        logger.error("transcribe.kill_handler_failed", error=str(e))


_asr: ASRModel | None = None


def reap_stale_jobs() -> int:
    """Fail jobs stuck in `processing` past the budget for their length.

    Called at worker startup to clean up rows orphaned by a previous
    worker that was killed mid-job. Each job is judged against
    `stale_cutoff_for(its own duration)`, so a long recording is not
    mistaken for a dead one. Returns the number of jobs reaped.
    """
    now = datetime.now(UTC)
    db = SessionLocal()
    try:
        processing = db.query(Job).filter(Job.status == "processing").all()
        reaped = 0
        for job in processing:
            if job.started_at is None:
                continue
            if now - job.started_at <= stale_cutoff_for(job.duration):
                continue
            job.status = "failed"
            job.error_message = "worker terminated before completion"
            job.finished_at = now
            reaped += 1
        db.commit()
        return reaped
    finally:
        db.close()


def index_chunks(db, transcript_id: str, segments: list[TranscriptSegment]) -> int:
    """Chunk, embed and persist a transcript's segments for search.

    Splits the Whisper segments into overlapping time windows, embeds all
    of them in a single batch (much faster than one call per chunk), and
    adds a `Chunk` row per window. The rows are added to `db` but not
    committed, the caller commits them together with the job's final
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
    for chunk, embedding in zip(chunks, embeddings, strict=True):
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
        if job is None:
            # The queue and the database can disagree: Redis holds the job,
            # Postgres holds the row, and nothing ties the two lifetimes
            # together. A queue that outlives its database (a recreated
            # volume, a restored dump) hands the worker an id with no row
            # behind it. There is nothing to transcribe and nothing to mark
            # failed, so say so and stop; the `finally` still removes the
            # spooled file. Raising would only turn a stale queue entry into
            # a traceback. The failure handler below has always guarded this
            # case; the success path had not.
            logger.warning("transcribe.orphaned", reason="job row not found")
            return
        job.status = "processing"
        job.started_at = datetime.now(UTC)
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
        job.finished_at = datetime.now(UTC)
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
            job.finished_at = datetime.now(UTC)
            db.commit()
        logger.error("transcribe.failed", error=str(e))
        raise
    finally:
        db.close()
        if os.path.exists(file_path):
            os.remove(file_path)
