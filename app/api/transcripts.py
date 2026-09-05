"""HTTP routes for reading the corpus itself, rather than searching it.

Two capabilities that the API did not have: *what is in here* and *what is
said around a given point*. They exist as ordinary REST endpoints first,
and the MCP tools consume them through the same shared functions, so the
browser console and the Streamlit demo can use them without speaking MCP,
and the two front ends cannot answer the same question differently.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import get_api_key
from app.core.logging import logger
from app.core.retrieval import (
    MAX_WINDOW_SEC,
    TranscriptSummary,
    TranscriptWindow,
    list_curated_transcripts,
    transcript_window,
)
from app.db.models import ApiKey
from app.db.session import SessionLocal

router = APIRouter()


@router.get("/transcripts", response_model=list[TranscriptSummary])
def list_transcripts(api_key: ApiKey = Depends(get_api_key)):
    """List the curated corpus: what is in here?

    Deliberately **not** the whole index. `/search` ranks everything
    stored, including visitor uploads; this listing shows only the curated
    demo corpus, because a listing surfaces every row it can see whether
    or not it is relevant. The reasoning, and the fact that the marker
    behind it is an expedient rather than a visibility model, is in
    `app.core.retrieval.list_curated_transcripts`.

    Args:
        api_key: Authenticated caller, resolved from the `X-API-Key`
            header. Any valid key sees the same corpus: this is not
            scoped to the caller.

    Returns:
        One entry per curated transcript with its id, source filename,
        duration in seconds and as mm:ss, language, and how many indexed
        passages it holds.

    Raises:
        HTTPException: 401 if the `X-API-Key` header is missing or
            invalid.
    """
    db = SessionLocal()
    try:
        transcripts = list_curated_transcripts(db)
    finally:
        db.close()

    logger.info("transcripts.listed", api_key_hash=api_key.key_hash, transcripts=len(transcripts))
    return transcripts


@router.get("/transcripts/{transcript_id}/window", response_model=TranscriptWindow)
def read_transcript_window(
    transcript_id: str,
    start_sec: float = Query(ge=0, description="Start of the interval to read, in seconds."),
    end_sec: float = Query(gt=0, description=f"End of the interval, at most {MAX_WINDOW_SEC:.0f}s after the start."),
    api_key: ApiKey = Depends(get_api_key),
):
    """Read a transcript over a time interval: what is said around a point.

    This is what turns a search hit into a quotation. A hit covers about
    45 seconds, which is often half of the sentence somebody wants to
    quote; this reads the text either side of it.

    Not scoped to the caller's key, matching `/search`: any valid key can
    read any indexed transcript.

    Args:
        transcript_id: Which transcript to read.
        start_sec: Start of the interval, in seconds.
        end_sec: End of the interval, in seconds.
        api_key: Authenticated caller, resolved from the `X-API-Key`
            header.

    Returns:
        The text over the interval and the span it actually covers, which
        may be slightly wider than requested because segments are
        returned whole rather than cut mid-sentence.

    Raises:
        HTTPException: 401 if the `X-API-Key` header is missing or
            invalid; 400 if the interval is empty, inverted or wider than
            the cap, which is refused rather than quietly trimmed; 404 if
            no such transcript exists.
    """
    db = SessionLocal()
    try:
        window = transcript_window(db, transcript_id, start_sec, end_sec)
    except ValueError as invalid:
        raise HTTPException(400, str(invalid)) from invalid
    except LookupError as missing:
        raise HTTPException(404, "transcript not found") from missing
    finally:
        db.close()

    logger.info(
        "transcripts.window_read",
        api_key_hash=api_key.key_hash,
        transcript_id=transcript_id,
        start_sec=start_sec,
        end_sec=end_sec,
        characters=len(window.text),
    )
    return window
