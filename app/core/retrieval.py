"""Retrieval over indexed transcripts, shared by every front end.

The query logic lives here rather than inside a route handler because it
has more than one caller: the HTTP API and the MCP server both answer the
same three questions ("what is in this corpus?", "where is X discussed?",
"what exactly is said around that point?"), and two copies of a ranking
rule drift apart silently. `_owned_job` in `app/api/jobs.py` is the same
move for the same reason.

Every function takes a `Session` and never opens one: opening a session is
the caller's job, and passing one in is what makes these testable without
Postgres. They are written with `select()` statements executed through the
session, so a test can capture the statement and read the rules off it,
rather than having to trust a fake that only replays rows.

Nothing here imports FastAPI. An invalid argument raises `ValueError`, and
each front end turns that into whatever its protocol calls an error.
"""

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.models import Chunk


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


def search_chunks(
    db,
    query_embedding: list[float],
    top_k: int,
    transcript_id: str | None = None,
) -> list[SearchHit]:
    """Rank indexed chunks against an embedded query by cosine distance.

    The ordering and the `LIMIT` both happen inside Postgres, so only
    `top_k` rows cross the wire and nothing is re-sorted here.

    The search covers **the whole index**: it is not scoped to the caller's
    key, and not restricted to the curated corpus the way
    `list_transcripts` is. That asymmetry is deliberate and explained in
    `list_curated_transcripts`.

    Args:
        db: Open session. Not opened or closed here.
        query_embedding: The query vector, from the same model that
            embedded the chunks at index time.
        top_k: Maximum number of hits to return.
        transcript_id: Restrict the search to a single transcript.
            Searches everything indexed when omitted.

    Returns:
        Hits ordered from most to least similar. An empty list means
        nothing is indexed (or nothing matched the filter): retrieval
        always returns the nearest neighbours however far away, so a low
        top score is the signal that a query is out of distribution, not
        an empty result.
    """
    distance = Chunk.embedding.cosine_distance(query_embedding)

    statement = select(Chunk, distance.label("distance"))
    if transcript_id is not None:
        statement = statement.where(Chunk.transcript_id == transcript_id)
    rows = db.execute(statement.order_by(distance).limit(top_k)).all()

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
