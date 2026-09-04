"""Per-API-key audio rate limiting.

Each `ApiKey` carries a `daily_minute_quota`: the total seconds of audio
it may submit within a rolling 24-hour window. Because the quota is
measured in audio seconds (not request count), the incoming file's
duration must be known *before* the job is accepted, so long uploads are
rejected up front rather than after the worker has spent CPU on them.

`probe_duration` reads that duration from the container metadata with
ffprobe; `enforce_quota` sums the durations already consumed in the
window and raises HTTP 429 if the new upload would push the key over.
"""

import subprocess
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import ApiKey, Job

RATE_LIMIT_WINDOW = timedelta(hours=24)


def probe_duration(path: str) -> float:
    """Return the media duration of `path` in seconds via ffprobe.

    Reads only the container metadata, so it is fast and does not decode
    the stream. This blocks (it shells out to ffprobe); call it through a
    threadpool from async request handlers.

    Raises:
        HTTPException: 400 if ffprobe cannot determine a duration (e.g.
            the file is corrupt or not actually a media file, the upload
            extension check is by filename only, so this is where bogus
            content is caught).
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return float(proc.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        raise HTTPException(400, "could not read media duration") from e


def used_seconds(db: Session, key_hash: str) -> float:
    """Sum the audio seconds a key has consumed in the rolling window.

    Counts every non-failed job created within `RATE_LIMIT_WINDOW`;
    failed jobs are excluded so callers are not charged for service
    errors. Jobs with no recorded duration contribute zero.
    """
    window_start = datetime.now(UTC) - RATE_LIMIT_WINDOW
    return db.query(func.coalesce(func.sum(Job.duration), 0.0)).filter(
        Job.api_key_hash == key_hash,
        Job.created_at >= window_start,
        Job.status != "failed",
    ).scalar()


def enforce_quota(db: Session, api_key: ApiKey, incoming_seconds: float) -> None:
    """Reject the current upload if it would exceed the key's quota.

    Args:
        db: Open session used for the usage query.
        api_key: The authenticated key whose `daily_minute_quota` applies.
        incoming_seconds: Duration of the upload being submitted.

    Raises:
        HTTPException: 429 if the seconds already used in the window plus
            `incoming_seconds` exceed the quota.
    """
    quota_seconds = api_key.daily_minute_quota * 60
    used = used_seconds(db, api_key.key_hash)
    if used + incoming_seconds > quota_seconds:
        remaining = max(0.0, quota_seconds - used)
        raise HTTPException(
            429,
            detail=(
                f"daily audio quota exceeded: {used:.0f}s of {quota_seconds}s "
                f"used in the last 24h ({remaining:.0f}s remaining); "
                f"this upload is {incoming_seconds:.0f}s"
            ),
        )
