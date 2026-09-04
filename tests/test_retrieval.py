"""Tests for the retrieval functions shared by the HTTP and MCP front ends.

Same shape as `tests/test_job_ownership.py`: the functions take a session
instead of opening one, so a few lines of fake stand in for Postgres. What
they can prove is the mapping from rows to domain objects, and the shape of
the statement that would have been executed, which is where the rules that
matter live (the curated-corpus filter, the limit). What actually comes back
from pgvector belongs to `scripts/smoke_test.sh`.
"""

import pytest

from app.core.retrieval import (
    CURATED_CORPUS_MARKER,
    MAX_WINDOW_SEC,
    MIN_RELEVANT_SCORE,
    list_curated_transcripts,
    search_chunks,
    transcript_window,
)
from app.db.models import Chunk


class FakeResult:
    """The `Result` of a `Session.execute`, over preloaded rows."""

    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


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
    db = FakeSession([(chunk(text="orbital mechanics", start=12.0, end=42.0), 0.25, "nasa-apollo11.mp3")])

    (hit,) = search_chunks(db, EMBEDDING, top_k=5)

    assert hit.transcript_id == "transcript-1"
    assert hit.text == "orbital mechanics"
    assert (hit.start_sec, hit.end_sec) == (12.0, 42.0)


def test_the_score_is_the_complement_of_the_cosine_distance():
    """pgvector returns a distance; callers rank on similarity."""
    db = FakeSession([(chunk(), 0.25, "nasa-apollo11.mp3")])

    (hit,) = search_chunks(db, EMBEDDING, top_k=5)

    assert hit.score == 0.75


def test_the_database_ordering_is_preserved():
    """Ordering and the limit both happen in Postgres; nothing re-sorts here."""
    rows = [(chunk(chunk_index=i), 0.1 * i, "nasa-apollo11.mp3") for i in range(3)]

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


def test_a_hit_names_the_recording_it_came_from():
    """A UUID is not a citation, and the filename is two joins away.

    Nothing consuming a hit can make those joins itself: not the model
    calling the tool, not the page rendering the result.
    """
    db = FakeSession([(chunk(), 0.25, "nasa-apollo11.mp3")])

    (hit,) = search_chunks(db, EMBEDDING, top_k=5)

    assert hit.audio_filename == "nasa-apollo11.mp3"


def test_a_hit_carries_quotable_timestamps():
    """"at 1094.3 seconds" is not something a reader can go and check."""
    db = FakeSession([(chunk(start=1094.3, end=1140.0), 0.25, "nasa-apollo11.mp3")])

    (hit,) = search_chunks(db, EMBEDDING, top_k=5)

    assert (hit.start, hit.end) == ("18:14", "19:00")


def test_a_weak_hit_is_marked_and_kept():
    """Filtering silently would tell the caller the corpus holds nothing.

    Retrieval always returns the nearest neighbours however far away, so a
    weak hit is not an error: it is the answer "not found, and this was the
    closest". Removing it turns that into "the corpus is empty on this
    subject", which is a different and usually false claim.
    """
    distance_of_a_weak_hit = 1.0 - (MIN_RELEVANT_SCORE - 0.05)
    db = FakeSession([(chunk(), distance_of_a_weak_hit, "nasa-apollo11.mp3")])

    (hit,) = search_chunks(db, EMBEDDING, top_k=5)

    assert hit.below_threshold is True
    assert hit.score < MIN_RELEVANT_SCORE


def test_a_strong_hit_is_not_marked():
    db = FakeSession([(chunk(), 0.2, "nasa-apollo11.mp3")])

    (hit,) = search_chunks(db, EMBEDDING, top_k=5)

    assert hit.below_threshold is False


def test_the_threshold_is_a_boundary_not_a_gap():
    """Exactly at the threshold counts as relevant: it is a floor to clear."""
    db = FakeSession([(chunk(), 1.0 - MIN_RELEVANT_SCORE, "nasa-apollo11.mp3")])

    (hit,) = search_chunks(db, EMBEDDING, top_k=5)

    assert hit.score == pytest.approx(MIN_RELEVANT_SCORE)
    assert hit.below_threshold is False


def segment(start: float, end: float, text: str) -> dict:
    """A row of `Transcript.word_timestamps`, as the worker writes it."""
    return {"start": start, "end": end, "text": text, "words": []}


SEGMENTS = [
    segment(0.0, 10.0, "Good morning."),
    segment(10.0, 20.0, "Today we talk about the lunar lander."),
    segment(20.0, 30.0, "It launched in July."),
    segment(30.0, 40.0, "And that is all."),
]


def a_transcript(segments=None, filename="nasa-apollo11.mp3"):
    return FakeSession([(segments if segments is not None else SEGMENTS, filename)])


def test_a_window_returns_the_text_over_the_interval():
    window = transcript_window(a_transcript(), "transcript-1", 10.0, 30.0)

    assert window.text == "Today we talk about the lunar lander. It launched in July."


def test_a_window_carries_what_a_citation_needs():
    window = transcript_window(a_transcript(), "transcript-1", 10.0, 30.0)

    assert window.transcript_id == "transcript-1"
    assert window.audio_filename == "nasa-apollo11.mp3"
    assert (window.start, window.end) == ("0:10", "0:30")


def test_a_segment_straddling_the_edge_is_included_whole():
    """Segments are not cut: half a sentence is worse than a second too much."""
    window = transcript_window(a_transcript(), "transcript-1", 15.0, 25.0)

    assert window.text == "Today we talk about the lunar lander. It launched in July."
    assert (window.start_sec, window.end_sec) == (10.0, 30.0)


def test_the_covered_span_is_reported_not_the_requested_one():
    window = transcript_window(a_transcript(), "transcript-1", 15.0, 25.0)

    assert (window.start_sec, window.end_sec) == (10.0, 30.0)


def test_an_interval_past_the_end_of_the_recording_comes_back_empty():
    """Empty, not an error: "nothing is said there" is a real answer."""
    window = transcript_window(a_transcript(), "transcript-1", 300.0, 400.0)

    assert window.text == ""
    assert (window.start_sec, window.end_sec) == (300.0, 400.0)


def test_a_window_wider_than_the_cap_is_refused_and_says_so():
    """It must not answer with a quietly truncated window.

    Without a cap this is a way to pull a whole transcript through one
    tool call, and an oversized response would be cut by the client
    anyway - at a point nobody chose, with nothing said about it.
    """
    with pytest.raises(ValueError) as refusal:
        transcript_window(a_transcript(), "transcript-1", 0.0, MAX_WINDOW_SEC + 1)

    assert f"{MAX_WINDOW_SEC:.0f}s cap" in str(refusal.value)


def test_a_window_exactly_at_the_cap_is_allowed():
    """The cap is a maximum, not a threshold to stay under."""
    assert transcript_window(a_transcript(), "transcript-1", 0.0, MAX_WINDOW_SEC) is not None


def test_an_interval_that_ends_before_it_starts_is_refused():
    with pytest.raises(ValueError):
        transcript_window(a_transcript(), "transcript-1", 30.0, 10.0)


def test_an_unknown_transcript_is_a_lookup_failure_not_an_empty_window():
    """An empty window would read as "nothing is said there", which is a lie."""
    with pytest.raises(LookupError):
        transcript_window(FakeSession([]), "no-such-transcript", 0.0, 10.0)
