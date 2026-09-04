"""Tests for the retrieval functions shared by the HTTP and MCP front ends.

Same shape as `tests/test_job_ownership.py`: the functions take a session
instead of opening one, so a few lines of fake stand in for Postgres. What
they can prove is the mapping from rows to domain objects, and the shape of
the statement that would have been executed, which is where the rules that
matter live (the curated-corpus filter, the limit). What actually comes back
from pgvector belongs to `scripts/smoke_test.sh`.
"""

from app.core.retrieval import CURATED_CORPUS_MARKER, list_curated_transcripts, search_chunks
from app.db.models import Chunk


class FakeResult:
    """The `Result` of a `Session.execute`, over preloaded rows."""

    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeSession:
    """Stands in for a Session: records the statement, returns fixed rows."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return FakeResult(self.rows)


def chunk(chunk_index: int = 0, text: str = "a passage", start: float = 0.0, end: float = 30.0) -> Chunk:
    return Chunk(
        id=f"chunk-{chunk_index}",
        transcript_id="transcript-1",
        chunk_index=chunk_index,
        start_sec=start,
        end_sec=end,
        text=text,
    )


EMBEDDING = [0.0] * 384


def test_a_row_becomes_a_hit_carrying_its_time_span():
    db = FakeSession([(chunk(text="orbital mechanics", start=12.0, end=42.0), 0.25)])

    (hit,) = search_chunks(db, EMBEDDING, top_k=5)

    assert hit.transcript_id == "transcript-1"
    assert hit.text == "orbital mechanics"
    assert (hit.start_sec, hit.end_sec) == (12.0, 42.0)


def test_the_score_is_the_complement_of_the_cosine_distance():
    """pgvector returns a distance; callers rank on similarity."""
    db = FakeSession([(chunk(), 0.25)])

    (hit,) = search_chunks(db, EMBEDDING, top_k=5)

    assert hit.score == 0.75


def test_the_database_ordering_is_preserved():
    """Ordering and the limit both happen in Postgres; nothing re-sorts here."""
    rows = [(chunk(chunk_index=i), 0.1 * i) for i in range(3)]

    hits = search_chunks(FakeSession(rows), EMBEDDING, top_k=3)

    assert [hit.chunk_index for hit in hits] == [0, 1, 2]


def test_the_limit_is_pushed_into_the_statement():
    """`top_k` must bound the query, not a slice taken after the fact."""
    db = FakeSession()

    search_chunks(db, EMBEDDING, top_k=7)

    assert "LIMIT" in str(db.statement)
    assert db.statement.compile().params["param_1"] == 7


def test_a_transcript_filter_reaches_the_statement():
    db = FakeSession()

    search_chunks(db, EMBEDDING, top_k=5, transcript_id="transcript-9")

    assert "transcript-9" in str(db.statement.compile(compile_kwargs={"literal_binds": True}))


def test_without_a_filter_the_search_covers_the_whole_index():
    """Search is deliberately unscoped: no key filter, no corpus filter."""
    db = FakeSession()

    search_chunks(db, EMBEDDING, top_k=5)

    assert "WHERE" not in str(db.statement)


def test_the_listing_is_restricted_to_the_curated_corpus():
    """Listing is indiscriminate by definition, so it must be filtered.

    A stranger's upload turns up in a *search* only if it is relevant, but
    in a *listing* it turns up always. With public upload on, that is the
    first thing a reviewer would be read back.
    """
    db = FakeSession()

    list_curated_transcripts(db)

    sql = str(db.statement.compile(compile_kwargs={"literal_binds": True}))
    assert f"'{CURATED_CORPUS_MARKER}'" in sql


def test_a_listed_transcript_carries_what_a_reader_recognises():
    db = FakeSession([("transcript-1", "nasa-apollo11.mp3", 1140.5, "en", 34)])

    (entry,) = list_curated_transcripts(db)

    assert entry.transcript_id == "transcript-1"
    assert entry.audio_filename == "nasa-apollo11.mp3"
    assert entry.language == "en"
    assert entry.passage_count == 34


def test_a_listed_duration_is_readable_as_well_as_numeric():
    """A model quoting "1140.5 seconds" is not quoting anything."""
    db = FakeSession([("transcript-1", "nasa-apollo11.mp3", 1140.5, "en", 34)])

    (entry,) = list_curated_transcripts(db)

    assert entry.duration_sec == 1140.5
    assert entry.duration == "19:00"


def test_a_transcript_with_no_duration_is_still_listable():
    """`Job.duration` is nullable: ffprobe can fail on a file that transcribes."""
    db = FakeSession([("transcript-1", "a.mp3", None, None, 3)])

    (entry,) = list_curated_transcripts(db)

    assert entry.duration_sec is None
    assert entry.duration is None
