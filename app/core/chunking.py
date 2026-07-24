"""Time-windowed chunking of Whisper segments for embedding.

Whisper emits many short segments (a few seconds each). For semantic
search we want coarser, overlapping windows of roughly a fixed duration,
so neighbouring context is preserved across chunk boundaries.

`chunk_segments` slides a time window over the segments: it accumulates
segments into a buffer and emits a chunk once the buffer spans
`target_duration` seconds, then re-seeds the buffer with the segments
falling in the last `overlap` seconds so the next chunk overlaps the
previous one. Chunks whose text is shorter than `MIN_CHUNK_CHARS` are
dropped as noise.

Segments may be mappings (e.g. `Transcript.word_timestamps` rows loaded
from the DB) or objects with `.start`/`.end`/`.text` (e.g. live
`TranscriptSegment`s); both are accepted.
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

MIN_CHUNK_CHARS = 50


def _field(seg: Any, name: str) -> Any:
    """Read `name` from a segment given as a mapping or an object."""
    if isinstance(seg, Mapping):
        return seg[name]
    return getattr(seg, name)


def _join_text(segs: list[Any]) -> str:
    """Concatenate segment texts, trimming Whisper's leading whitespace."""
    return " ".join(_field(s, "text").strip() for s in segs).strip()


def chunk_segments(
    segments: Iterable[Any],
    target_duration: float = 45.0,
    overlap: float = 10.0,
) -> Iterator[dict]:
    """Yield overlapping time-windowed chunks from Whisper segments.

    Args:
        segments: Ordered Whisper segments, each exposing `start`, `end`
            (seconds) and `text`.
        target_duration: Emit a chunk once the buffer spans at least this
            many seconds.
        overlap: Seconds of tail context carried into the next chunk.

    Yields:
        Dicts with `chunk_index` (contiguous, 0-based over emitted
        chunks), `start_sec`, `end_sec`, and `text`. Chunks whose text is
        shorter than `MIN_CHUNK_CHARS` are skipped.
    """
    buffer: list[Any] = []
    buffer_start: float | None = None
    chunk_index = 0
    last_end: float | None = None

    def emit(start: float, end: float) -> dict | None:
        nonlocal chunk_index
        text = _join_text(buffer)
        if len(text) < MIN_CHUNK_CHARS:
            return None
        chunk = {
            "chunk_index": chunk_index,
            "start_sec": start,
            "end_sec": end,
            "text": text,
        }
        chunk_index += 1
        return chunk

    for s in segments:
        if not buffer:
            buffer_start = _field(s, "start")
        buffer.append(s)
        s_end = _field(s, "end")

        if s_end - buffer_start >= target_duration:
            chunk = emit(buffer_start, s_end)
            if chunk is not None:
                yield chunk
            last_end = s_end

            # Re-seed the buffer with the tail overlapping the last
            # `overlap` seconds, so the next chunk overlaps this one.
            cutoff = s_end - overlap
            buffer = [seg for seg in buffer if _field(seg, "end") > cutoff]
            buffer_start = _field(buffer[0], "start") if buffer else None

    # Trailing buffer that never reached target_duration. Skip it if it is
    # only the carried-over overlap tail (adds nothing past the last chunk),
    # which would otherwise duplicate the previous chunk's ending.
    if buffer:
        trailing_end = _field(buffer[-1], "end")
        if last_end is None or trailing_end > last_end:
            chunk = emit(buffer_start, trailing_end)
            if chunk is not None:
                yield chunk
