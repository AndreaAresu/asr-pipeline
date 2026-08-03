"""Tests for the summarization helpers.

These cover the two things that do not need the LLM: how the transcript
is marked up before it is sent, and how the model's reply is bounded
before it is trusted.
"""

import re

from app.core.summarize import _sanitise_sections, format_transcript_for_llm, thin_segments

MARKER = re.compile(r"\[(\d+\.\d)s\]")


def segments(count: int, seg_duration: float = 5.0):
    return [
        {"start": i * seg_duration, "end": (i + 1) * seg_duration, "text": f"sentence {i}."}
        for i in range(count)
    ]


def test_first_marker_is_emitted_immediately():
    assert format_transcript_for_llm(segments(3)).startswith("[0.0s]")


def test_markers_are_spaced_by_the_interval():
    text = format_transcript_for_llm(segments(24), marker_interval=30.0)
    times = [float(t) for t in MARKER.findall(text)]
    assert times == sorted(times)
    for earlier, later in zip(times, times[1:], strict=False):
        assert later - earlier >= 30.0


def test_a_long_gap_does_not_emit_a_run_of_markers():
    """One marker after a silence, not one per interval skipped."""
    sparse = [
        {"start": 0.0, "end": 2.0, "text": "before the gap."},
        {"start": 600.0, "end": 602.0, "text": "after the gap."},
    ]
    assert len(MARKER.findall(format_transcript_for_llm(sparse))) == 2


def test_segment_text_is_preserved():
    text = format_transcript_for_llm(segments(3))
    assert "sentence 0." in text and "sentence 2." in text


def test_timestamps_beyond_the_audio_are_clamped():
    """The whole point of the guard: bound a hallucinated timestamp."""
    sections = _sanitise_sections(
        [{"title": "Invented", "start_sec": 9000, "end_sec": 99999, "key_points": ["x"]}],
        duration=120.0,
    )
    assert sections[0]["start_sec"] == 120.0
    assert sections[0]["end_sec"] == 120.0


def test_negative_timestamps_are_clamped_to_zero():
    sections = _sanitise_sections(
        [{"title": "Before time", "start_sec": -30, "end_sec": 10, "key_points": []}],
        duration=120.0,
    )
    assert sections[0]["start_sec"] == 0.0


def test_inverted_ranges_are_repaired():
    sections = _sanitise_sections(
        [{"title": "Backwards", "start_sec": 90, "end_sec": 30, "key_points": []}],
        duration=120.0,
    )
    assert sections[0]["start_sec"] == 30.0
    assert sections[0]["end_sec"] == 90.0


def test_sections_are_returned_in_chronological_order():
    sections = _sanitise_sections(
        [
            {"title": "Third", "start_sec": 80, "end_sec": 100, "key_points": []},
            {"title": "First", "start_sec": 0, "end_sec": 40, "key_points": []},
            {"title": "Second", "start_sec": 40, "end_sec": 80, "key_points": []},
        ],
        duration=120.0,
    )
    assert [s["title"] for s in sections] == ["First", "Second", "Third"]


def test_malformed_entries_are_dropped_not_fatal():
    sections = _sanitise_sections(
        ["a bare string", 42, None, {"title": "Good", "start_sec": 1, "end_sec": 2, "key_points": ["p"]}],
        duration=120.0,
    )
    assert [s["title"] for s in sections] == ["Good"]


def test_a_non_list_payload_yields_no_sections():
    assert _sanitise_sections({"not": "a list"}, duration=10.0) == []
    assert _sanitise_sections(None, duration=10.0) == []


def test_untitled_sections_get_a_placeholder():
    sections = _sanitise_sections([{"start_sec": 0, "end_sec": 5, "key_points": []}], duration=10.0)
    assert sections[0]["title"] == "Untitled section"


def long_segments(count: int, chars: int = 60):
    """Segments of a fixed text length, 5s apart — a stand-in for a long interview."""
    return [
        {"start": i * 5.0, "end": i * 5.0 + 5.0, "text": "x" * chars}
        for i in range(count)
    ]


def test_a_short_transcript_is_left_alone():
    segments = long_segments(10)
    kept, thinned = thin_segments(segments, max_chars=40_000)
    assert thinned is False
    assert kept == segments


def test_an_overlong_transcript_is_thinned_within_budget():
    kept, thinned = thin_segments(long_segments(2880), max_chars=40_000)
    assert thinned is True
    assert sum(len(s["text"]) for s in kept) <= 40_000


def test_thinning_preserves_coverage_of_the_whole_recording():
    """The failure this guards against: summarizing only the opening.

    Truncation would keep the first N characters and report the result as
    a summary of the whole thing. Thinning must still reach the end.
    """
    segments = long_segments(2880)
    kept, _ = thin_segments(segments, max_chars=40_000)
    total_span = segments[-1]["end"] - segments[0]["start"]
    kept_span = kept[-1]["end"] - kept[0]["start"]
    assert kept_span > total_span * 0.95


def test_thinning_an_empty_transcript_is_safe():
    assert thin_segments([], max_chars=100) == ([], False)


def test_section_count_is_capped():
    many = [{"title": f"S{i}", "start_sec": i, "end_sec": i + 1, "key_points": []} for i in range(20)]
    assert len(_sanitise_sections(many, duration=100.0)) == 5
