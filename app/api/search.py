"""HTTP route for semantic search over indexed transcripts.

Embeds the caller's query with the same model used to index chunks and
returns the nearest chunks by cosine distance, computed in Postgres by
pgvector. Results carry the time span they cover so a client can seek
straight to that point in the source audio.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.auth import get_api_key
from app.core.embeddings import embed_batch
from app.core.logging import logger
from app.db.models import ApiKey, Chunk
from app.db.session import SessionLocal

router = APIRouter()


class SearchRequest(BaseModel):
    """A semantic search query."""

    query: str = Field(description="Natural-language query to match against indexed transcript chunks.")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of hits to return.")
    transcript_id: str | None = Field(
        default=None,
        description="Restrict the search to a single transcript. Searches everything indexed when omitted.",
    )


class SearchHit(BaseModel):
    """One matching chunk of a transcript."""

    transcript_id: str = Field(description="Transcript the chunk belongs to.")
    chunk_index: int = Field(description="Position of the chunk within its transcript.")
    start_sec: float = Field(description="Start of the chunk in the source audio, in seconds.")
    end_sec: float = Field(description="End of the chunk in the source audio, in seconds.")
    text: str = Field(description="Chunk text.")
    score: float = Field(
        description="Cosine similarity to the query in [-1, 1]; 1 is identical. Computed as 1 - cosine_distance.",
    )


class SearchResponse(BaseModel):
    """Ranked hits for a search query."""

    query: str = Field(description="Echo of the submitted query.")
    hits: list[SearchHit] = Field(description="Hits ordered by descending score.")


def _search(query: str, top_k: int, transcript_id: str | None) -> list[SearchHit]:
    """Run the blocking embed + vector query and map rows to hits."""
    query_embedding = embed_batch([query])[0]

    db = SessionLocal()
    try:
        distance = Chunk.embedding.cosine_distance(query_embedding)
        stmt = db.query(Chunk, distance.label("distance"))
        if transcript_id is not None:
            stmt = stmt.filter(Chunk.transcript_id == transcript_id)
        rows = stmt.order_by(distance).limit(top_k).all()
    finally:
        db.close()

    return [
        SearchHit(
            transcript_id=chunk.transcript_id,
            chunk_index=chunk.chunk_index,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            text=chunk.text,
            score=1.0 - distance,
        )
        for chunk, distance in rows
    ]


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, api_key: ApiKey = Depends(get_api_key)):
    """Return the transcript chunks most semantically similar to a query.

    The query is embedded with the same sentence-transformers model used
    at index time, and ranked against stored chunk vectors by cosine
    distance inside Postgres, the ordering and the LIMIT both happen in
    the database, so only `top_k` rows come back.

    Args:
        request: Query text, result count, and optional transcript filter.
        api_key: Authenticated caller, resolved from the `X-API-Key`
            header.

    Returns:
        The echoed query and its hits, ordered from most to least
        similar. An empty `hits` list means nothing has been indexed yet
        (or nothing matched the transcript filter), retrieval itself
        always returns the nearest neighbours, however far away, so a low
        top score is the signal that a query is out of distribution.

    Raises:
        HTTPException: 401 if the `X-API-Key` header is missing or
            invalid.

    Note:
        Embedding and the vector query are both blocking, so they run in
        a threadpool to keep the event loop free.
    """
    hits = await run_in_threadpool(_search, request.query, request.top_k, request.transcript_id)

    logger.info(
        "search.executed",
        api_key_hash=api_key.key_hash,
        query=request.query,
        top_k=request.top_k,
        hits=len(hits),
        top_score=round(hits[0].score, 4) if hits else None,
    )
    return SearchResponse(query=request.query, hits=hits)
