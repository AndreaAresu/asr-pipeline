"""HTTP route for LLM summarization of a transcript.

Summaries are expensive (seconds of latency, tokens billed) and their
input never changes, so the result is cached in Postgres keyed by
transcript. A cache hit is a single primary-key lookup; only the first
call for a transcript reaches the LLM.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.auth import get_api_key
from app.core.logging import logger
from app.core.summarize import SummarizationUnavailable, summarize_transcript
from app.db.models import ApiKey, Job, Summary, Transcript
from app.db.session import SessionLocal

router = APIRouter()


class SummarySection(BaseModel):
    """One thematic section of a summarized transcript."""

    title: str = Field(description="Short label for the section, at most 8 words.")
    start_sec: float = Field(description="Start of the section in the source audio, in seconds.")
    end_sec: float = Field(description="End of the section in the source audio, in seconds.")
    key_points: list[str] = Field(description="Substantive points made during the section.")


class SummaryResponse(BaseModel):
    """A transcript summary and how it was produced."""

    transcript_id: str = Field(description="Transcript that was summarized.")
    cached: bool = Field(description="True when served from Postgres without calling the LLM.")
    model: str = Field(description="LLM that produced the summary.")
    sections: list[SummarySection] = Field(description="Thematic sections in chronological order.")
    meta: dict = Field(description="Model and token counts for the generating call, kept for cost auditing.")


def _generate(transcript_id: str) -> dict:
    """Load the transcript, summarize it, and persist the result.

    Runs entirely in a threadpool: both the DB work and the Groq call
    block. The transcript's audio duration is read from its owning job so
    section timestamps can be clamped to the real range.

    Raises:
        HTTPException: 404 if the transcript no longer exists.
    """
    db = SessionLocal()
    try:
        transcript = db.get(Transcript, transcript_id)
        if transcript is None:
            raise HTTPException(404, "transcript not found")

        segments = transcript.word_timestamps or []
        job = db.get(Job, transcript.job_id)
        duration = (job.duration if job else None) or (
            max((s["end"] for s in segments), default=0.0)
        )

        result = summarize_transcript(segments, duration)

        db.merge(
            Summary(
                transcript_id=transcript_id,
                summary_json=result,
                model_used=result["_meta"]["model"],
            )
        )
        db.commit()
        return result
    finally:
        db.close()


@router.post("/summarize/{transcript_id}", response_model=SummaryResponse)
async def summarize(transcript_id: str, api_key: ApiKey = Depends(get_api_key)):
    """Return a structured, timestamped summary of a transcript.

    On the first call for a transcript the text is sent to the LLM and
    the result stored; every later call for the same transcript is served
    from that stored row, so it costs one indexed lookup instead of a
    round trip and a few thousand tokens.

    Args:
        transcript_id: Id of the transcript to summarize — the `id` of a
            `Transcript`, not the `job_id` that produced it.
        api_key: Authenticated caller, resolved from the `X-API-Key`
            header.

    Returns:
        The summary sections with the time range each covers, whether it
        came from cache, and the token counts of the generating call.

    Raises:
        HTTPException: 401 if the `X-API-Key` header is missing or
            invalid; 404 if no such transcript exists; 503 if
            summarization is not configured (no `GROQ_API_KEY`).
    """
    db = SessionLocal()
    try:
        cached = db.get(Summary, transcript_id)
        payload = cached.summary_json if cached else None
        model_used = cached.model_used if cached else None
    finally:
        db.close()

    if payload is not None:
        logger.info("summarize.cache_hit", transcript_id=transcript_id, api_key_hash=api_key.key_hash)
    else:
        try:
            payload = await run_in_threadpool(_generate, transcript_id)
        except SummarizationUnavailable as e:
            logger.error("summarize.unavailable", transcript_id=transcript_id, error=str(e))
            raise HTTPException(503, str(e)) from e
        model_used = payload["_meta"]["model"]
        logger.info(
            "summarize.generated",
            transcript_id=transcript_id,
            api_key_hash=api_key.key_hash,
            sections=len(payload["sections"]),
            input_tokens=payload["_meta"]["input_tokens"],
            output_tokens=payload["_meta"]["output_tokens"],
        )

    return SummaryResponse(
        transcript_id=transcript_id,
        cached=cached is not None,
        model=model_used,
        sections=payload["sections"],
        meta=payload["_meta"],
    )
