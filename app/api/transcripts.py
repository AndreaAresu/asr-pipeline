"""HTTP routes for reading the corpus itself, rather than searching it.

Two capabilities that the API did not have: *what is in here* and *what is
said around a given point*. They exist as ordinary REST endpoints first,
and the MCP tools consume them through the same shared functions, so the
browser console and the Streamlit demo can use them without speaking MCP,
and the two front ends cannot answer the same question differently.
"""

from fastapi import APIRouter, Depends

from app.core.auth import get_api_key
from app.core.logging import logger
from app.core.retrieval import TranscriptSummary, list_curated_transcripts
from app.db.models import ApiKey
from app.db.session import SessionLocal

router = APIRouter()


@router.get("/transcripts", response_model=list[TranscriptSummary])
def list_transcripts(api_key: ApiKey = Depends(get_api_key)):
    """List the curated corpus, newest question first: what is in here?

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
