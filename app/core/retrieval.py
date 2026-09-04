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
from sqlalchemy import func, select

from app.core.chunking import segment_field
from app.db.models import Chunk, Job, Transcript

# The marker that identifies the curated corpus: the three NASA episodes
# restored from `data/seed/nasa_corpus.sql`, whose jobs carry this literal
# instead of the SHA-256 of a key.
#
# **This is an expedient, not a visibility model.** The literal exists
# because publishing a real key's hash in a public repo would be a leak
# (`scripts/dump_seed.sh` rewrites the column), and it happens to behave
# like a "part of the demo corpus" flag because it matches no real key.
# The schema has exactly one ownership column, `Job.api_key_hash`, and the
# domain does not distinguish *who uploaded* from *who may see*. Filtering
# by a real key instead would hide the demo corpus from everyone, since it
# belongs to nobody. The real question - an `is_public` flag? a corpus
# entity? - is open, and is written down under "Not yet specified" in
# `.scratch/mcp-server/map.md`. Do not build on this as if it were the
# answer.
#
# Practical consequence, worth knowing before wondering why a listing is
# empty: only a database restored from the seed dump has rows carrying this
# literal. A development stack that transcribed the same episodes itself has
# them under a real key hash, so `list_curated_transcripts` returns nothing
# there while `search_chunks` still finds every one of them.
CURATED_CORPUS_MARKER = "seed"

# Below this cosine similarity a hit is noise. Measured, not guessed, on the
# 109-chunk corpus (`scripts/search_eval.md`): 25 hits from five
# out-of-distribution queries topped out at 0.224, while every hit a human
# would keep sat at 0.372 or above, and 0.30 is the middle of that empty
# band. It expires with the corpus - the same queries scored *negative* on a
# single-chunk index - so re-run the evaluation before trusting it on a
# different one.
#
# It is **reported, never applied**: `search_chunks` marks a hit below it and
# returns it anyway. `demo/app.py` keeps its own copy of the number because
# it is a separate service that cannot import this package.
MIN_RELEVANT_SCORE = 0.30

# Widest transcript window that will be served, in seconds. Five minutes is
# an order of magnitude more than the ~45 seconds a search hit covers, and a
# fifth of an episode: enough to read around a passage, not enough to pull a
# whole transcript through repeated calls. It is also comfortably inside the
# response cap MCP clients impose, which is the other half of the reason -
# an oversized answer gets cut at a point nobody chose.
MAX_WINDOW_SEC = 300.0


def mmss(seconds: float | None) -> str | None:
    """Render seconds as `mm:ss`, or None for a missing duration.

    Timestamps exist here for humans and models to quote. Minutes are not
    wrapped into hours: "24:21" stays "24:21", because a reader scrubbing
    an audio player is looking for exactly that.
    """
    if seconds is None:
        return None
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


class SearchHit(BaseModel):
    """One matching passage of a transcript.

    A passage, not a document: these are the ~45-second chunks the
    transcript was split into for indexing. Everything a caller needs to
    quote one is on the object, because the two things that make it
    quotable - the recording's name and the time as a reader would write
    it - are otherwise two joins and a conversion away.
    """

    transcript_id: str = Field(description="Transcript the passage belongs to.")
    audio_filename: str = Field(description="Source recording this passage came from, e.g. 'nasa-apollo11.mp3'.")
    chunk_index: int = Field(description="Position of the passage within its transcript.")
    start_sec: float = Field(description="Start of the passage in the source audio, in seconds.")
    end_sec: float = Field(description="End of the passage in the source audio, in seconds.")
    start: str = Field(description="Start of the passage as mm:ss, for quoting.")
    end: str = Field(description="End of the passage as mm:ss, for quoting.")
    text: str = Field(description="What is said in the passage.")
    score: float = Field(
        description="Cosine similarity to the query in [-1, 1]; 1 is identical. Computed as 1 - cosine_distance.",
    )
    below_threshold: bool = Field(
        description=(
            "True when the score is under 0.30, the measured noise floor for this corpus. Such a hit is "
            "returned rather than dropped, because search always returns the nearest passages however far "
            "away: treat it as 'the closest thing in the index', not as an answer."
        ),
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
        Hits ordered from most to least similar, each marked with whether
        it clears the measured noise floor. An empty list means nothing is
        indexed (or nothing matched the filter): retrieval always returns
        the nearest neighbours however far away, so a low top score - not
        an empty result - is the signal that a query is out of
        distribution. Weak hits are marked and returned, never dropped.
    """
    distance = Chunk.embedding.cosine_distance(query_embedding)

    # The filename lives on `Job`, two joins from a chunk, and no caller can
    # make those joins for itself.
    statement = (
        select(Chunk, distance.label("distance"), Job.audio_filename)
        .join(Transcript, Chunk.transcript_id == Transcript.id)
        .join(Job, Transcript.job_id == Job.id)
    )
    if transcript_id is not None:
        statement = statement.where(Chunk.transcript_id == transcript_id)
    rows = db.execute(statement.order_by(distance).limit(top_k)).all()

    return [
        SearchHit(
            transcript_id=chunk.transcript_id,
            audio_filename=audio_filename,
            chunk_index=chunk.chunk_index,
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            start=mmss(chunk.start_sec),
            end=mmss(chunk.end_sec),
            text=chunk.text,
            score=1.0 - distance,
            below_threshold=(1.0 - distance) < MIN_RELEVANT_SCORE,
        )
        for chunk, distance, audio_filename in rows
    ]


class TranscriptSummary(BaseModel):
    """One transcript in the curated corpus, as an index entry."""

    transcript_id: str = Field(description="Pass this to a search filter or to a transcript window.")
    audio_filename: str = Field(description="Source recording, e.g. 'nasa-apollo11.mp3'.")
    duration_sec: float | None = Field(description="Length of the recording in seconds; null if it was never measured.")
    duration: str | None = Field(description="The same length as mm:ss, for quoting.")
    language: str | None = Field(description="Language detected by Whisper, as an ISO 639-1 code.")
    passage_count: int = Field(description="Number of indexed passages this transcript was split into.")


def list_curated_transcripts(db) -> list[TranscriptSummary]:
    """List the curated corpus: what a reader can expect to find here.

    Restricted to `CURATED_CORPUS_MARKER`, unlike `search_chunks`, which
    covers the whole index. The asymmetry is the point: an unrelated
    upload surfaces in a *search* only when it is semantically relevant,
    but it surfaces in a *listing* always, because listing is
    indiscriminate by definition. With public upload open on the demo,
    that would make a stranger's filename the first thing this project
    says about itself.

    Args:
        db: Open session. Not opened or closed here.

    Returns:
        One entry per transcript, ordered by filename so the listing is
        stable between calls.
    """
    statement = (
        select(
            Transcript.id,
            Job.audio_filename,
            Job.duration,
            Transcript.language,
            func.count(Chunk.id),
        )
        .join(Job, Transcript.job_id == Job.id)
        .outerjoin(Chunk, Chunk.transcript_id == Transcript.id)
        .where(Job.api_key_hash == CURATED_CORPUS_MARKER)
        .group_by(Transcript.id, Job.audio_filename, Job.duration, Transcript.language)
        .order_by(Job.audio_filename)
    )

    return [
        TranscriptSummary(
            transcript_id=transcript_id,
            audio_filename=audio_filename,
            duration_sec=duration,
            duration=mmss(duration),
            language=language,
            passage_count=passage_count,
        )
        for transcript_id, audio_filename, duration, language, passage_count in db.execute(statement).all()
    ]


class TranscriptWindow(BaseModel):
    """The text of one transcript over a time interval."""

    transcript_id: str = Field(description="Transcript the text was read from.")
    audio_filename: str = Field(description="Source recording, e.g. 'nasa-apollo11.mp3'.")
    start_sec: float = Field(description="Start of the text actually returned, in seconds.")
    end_sec: float = Field(description="End of the text actually returned, in seconds.")
    start: str = Field(description="The same start as mm:ss, for quoting.")
    end: str = Field(description="The same end as mm:ss, for quoting.")
    text: str = Field(
        description=(
            "What is said over the interval, whole segments only, so the quote does not begin or end "
            "mid-sentence. Empty when nothing is said there, which usually means the interval is past "
            "the end of the recording."
        ),
    )


def transcript_window(db, transcript_id: str, start_sec: float, end_sec: float) -> TranscriptWindow:
    """Read one transcript over a time interval, for quoting around a hit.

    Built from the stored Whisper segments rather than from chunks:
    chunks overlap by ten seconds, so stitching them would repeat text.
    Segments are contiguous, and are returned whole - a window that
    started mid-sentence would be a worse quote than one a second too
    wide.

    Reading is not scoped to the caller's key, matching `/search` and
    `/summarize`: any valid key can read any indexed transcript. Job
    *status* is key-scoped, transcript *content* is not, which is the
    existing shape of this system and not something this function decides.

    Args:
        db: Open session. Not opened or closed here.
        transcript_id: Which transcript to read.
        start_sec: Start of the interval, in seconds.
        end_sec: End of the interval, in seconds.

    Returns:
        The text over the interval, with the span it actually covers -
        which can be slightly wider than what was asked for, because
        segments are returned whole.

    Raises:
        ValueError: If the interval is empty or inverted, or wider than
            `MAX_WINDOW_SEC`. Refused rather than truncated: a caller can
            ask again for less, but cannot notice a silent trim.
        LookupError: If no such transcript exists. An empty window would
            read as "nothing is said there", which is a different claim.
    """
    if end_sec <= start_sec:
        raise ValueError(f"end_sec ({end_sec}) must be greater than start_sec ({start_sec})")
    if end_sec - start_sec > MAX_WINDOW_SEC:
        raise ValueError(
            f"window of {end_sec - start_sec:.0f}s is wider than the {MAX_WINDOW_SEC:.0f}s cap; "
            "ask for a narrower interval"
        )

    row = db.execute(
        select(Transcript.word_timestamps, Job.audio_filename)
        .join(Job, Transcript.job_id == Job.id)
        .where(Transcript.id == transcript_id)
    ).first()
    if row is None:
        raise LookupError(f"no transcript {transcript_id}")
    segments, audio_filename = row

    covered = [
        seg
        for seg in segments
        if segment_field(seg, "end") > start_sec and segment_field(seg, "start") < end_sec
    ]

    covered_start = segment_field(covered[0], "start") if covered else start_sec
    covered_end = segment_field(covered[-1], "end") if covered else end_sec

    return TranscriptWindow(
        transcript_id=transcript_id,
        audio_filename=audio_filename,
        start_sec=covered_start,
        end_sec=covered_end,
        start=mmss(covered_start),
        end=mmss(covered_end),
        text=" ".join(segment_field(seg, "text").strip() for seg in covered).strip(),
    )
