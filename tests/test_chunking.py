"""Tests for time-windowed chunking of Whisper segments."""

import pytest

from app.core.chunking import MIN_CHUNK_CHARS, chunk_segments


def make_segments(count: int, seg_duration: float = 5.0, text: str | None = None):
    """Build `count` contiguous segments of `seg_duration` seconds each."""
    body = text if text is not None else "This is a reasonably long sentence of transcript text."
    return [
        {"start": i * seg_duration, "end": (i + 1) * seg_duration, "text": body}
        for i in range(count)
    ]


def test_no_segments_yields_nothing():
    assert list(chunk_segments([])) == []


def test_short_audio_emits_a_single_trailing_chunk():
    # 20s of audio never reaches the 45s window, but must still be indexed.
    chunks = list(chunk_segments(make_segments(4)))
    assert len(chunks) == 1
    assert chunks[0]["start_sec"] == 0.0
    assert chunks[0]["end_sec"] == 20.0


def test_window_closes_at_target_duration():
    chunks = list(chunk_segments(make_segments(20), target_duration=45.0, overlap=10.0))
    assert len(chunks) > 1
    first = chunks[0]
    # The window closes on the first segment that reaches the target, so the
    # chunk spans at least target_duration and at most one segment more.
    assert 45.0 <= first["end_sec"] - first["start_sec"] <= 50.0


def test_consecutive_chunks_overlap_and_never_gap():
    chunks = list(chunk_segments(make_segments(40), target_duration=45.0, overlap=10.0))
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current["start_sec"] < previous["end_sec"], "chunks must overlap, not gap"


def test_chunk_indices_are_contiguous_from_zero():
    chunks = list(chunk_segments(make_segments(40)))
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_short_text_is_filtered_as_noise():
    # Ten minutes of "uh" is silence-adjacent noise, not content.
    chunks = list(chunk_segments(make_segments(40, text="uh")))
    assert chunks == []


def test_emitted_text_meets_the_minimum_length():
    chunks = list(chunk_segments(make_segments(40)))
    assert all(len(c["text"]) >= MIN_CHUNK_CHARS for c in chunks)


def test_trailing_overlap_tail_is_not_re_emitted():
    """The leftover buffer after the last chunk is pure overlap.

    Emitting it would duplicate the previous chunk's ending as a new
    chunk that covers no new audio.
    """
    segments = make_segments(20, seg_duration=5.0)
    chunks = list(chunk_segments(segments, target_duration=45.0, overlap=10.0))
    audio_end = segments[-1]["end"]
    assert chunks[-1]["end_sec"] == audio_end


class Segment:
    """Object-style segment, as produced live by the ASR wrapper."""

    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


def test_objects_and_mappings_are_both_accepted():
    """Segments arrive as objects from the model and as dicts from JSONB."""
    as_dicts = make_segments(20)
    as_objects = [Segment(s["start"], s["end"], s["text"]) for s in as_dicts]
    assert list(chunk_segments(as_dicts)) == list(chunk_segments(as_objects))


@pytest.mark.parametrize("target", [30.0, 45.0, 60.0])
def test_all_audio_is_covered_for_any_window_size(target):
    segments = make_segments(60)
    chunks = list(chunk_segments(segments, target_duration=target, overlap=10.0))
    assert chunks[0]["start_sec"] == segments[0]["start"]
    assert chunks[-1]["end_sec"] == segments[-1]["end"]
