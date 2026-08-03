"""LLM summarization of transcripts into timestamped sections.

Sends a transcript to Groq (Llama 3.3 70B) and gets back a structured
JSON breakdown: 3-5 thematic sections, each with a title, a time range,
and key points.

The hard part is making the timestamps *real*. An LLM asked to summarize
plain prose will happily invent plausible-looking times. Two defences,
both needed:

1. The transcript handed to the model is interleaved with explicit
   `[123.4s]` markers every `MARKER_INTERVAL` seconds, so the model is
   copying timestamps it can see rather than estimating them.
2. Whatever comes back is clamped to the transcript's real duration and
   forced to be ordered (`_sanitise_sections`), so a hallucinated value
   is bounded instead of surfacing to the client.

Token usage is recorded on every call under the `_meta` key. Knowing the
marginal cost of a request per tenant is a production requirement, not a
nicety, and it is far easier to record from the start than to backfill.
"""

import json
from typing import Any

from groq import Groq

from app.config import settings
from app.core.chunking import segment_field
from app.core.logging import logger

MARKER_INTERVAL = 30.0
MAX_SECTIONS = 5

SYSTEM_PROMPT = """\
You are an expert at summarizing conversation transcripts.

You receive a transcript annotated with timestamp markers in the form \
[123.4s], which give the elapsed time in seconds at that point in the audio.

Produce a JSON object with a single key "sections", holding 3 to 5 thematic \
sections in chronological order. Each section is an object with:
  - "title": a short label, at most 8 words
  - "start_sec": float, the timestamp where the theme begins
  - "end_sec": float, the timestamp where the theme ends
  - "key_points": a list of 2 to 4 short strings, the substantive points made

Rules:
  - Take start_sec and end_sec ONLY from the [Ns] markers you can see in the \
transcript. Never invent a timestamp, never estimate one.
  - Sections must not overlap, and must cover the transcript in order.
  - Key points state what was actually said. Do not add outside knowledge.
  - Return ONLY the JSON object, with no surrounding prose or code fences.\
"""

_client: Groq | None = None


class SummarizationUnavailable(RuntimeError):
    """Raised when summarization is requested without a configured LLM key."""


def get_client() -> Groq:
    """Return the lazily-constructed, process-wide Groq client.

    Raises:
        SummarizationUnavailable: If `groq_api_key` is not configured.
            The rest of the service (transcribe, search) runs fine
            without it, so this is deliberately a runtime failure of one
            endpoint rather than a startup failure of the process.
    """
    global _client
    if not settings.groq_api_key:
        raise SummarizationUnavailable(
            "GROQ_API_KEY is not configured; summarization is disabled"
        )
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def format_transcript_for_llm(segments: list[Any], marker_interval: float = MARKER_INTERVAL) -> str:
    """Render segments as text with periodic timestamp markers.

    A `[<start>s]` marker is emitted before the first segment and then
    whenever `marker_interval` seconds have elapsed since the last
    marker, giving the model concrete timestamps to quote back.

    Args:
        segments: Ordered Whisper segments, as mappings or objects with
            `start`/`end`/`text`.
        marker_interval: Seconds between markers.

    Returns:
        A single string of marked-up transcript text.
    """
    parts: list[str] = []
    next_marker = 0.0

    for seg in segments:
        start = segment_field(seg, "start")
        if start >= next_marker:
            parts.append(f"[{start:.1f}s]")
            # Skip ahead past any intervals this segment jumped over, so a
            # long gap does not emit a run of consecutive markers.
            next_marker = (int(start / marker_interval) + 1) * marker_interval
        parts.append(segment_field(seg, "text").strip())

    return " ".join(parts)


def _sanitise_sections(sections: Any, duration: float) -> list[dict]:
    """Coerce the model's sections into a well-formed, in-range list.

    Drops anything malformed, clamps times into `[0, duration]`, repairs
    inverted ranges, and truncates to `MAX_SECTIONS`. This is the guard
    that keeps a hallucinated timestamp from reaching the client as if it
    were real.
    """
    if not isinstance(sections, list):
        return []

    cleaned: list[dict] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        try:
            start = min(max(float(section.get("start_sec", 0.0)), 0.0), duration)
            end = min(max(float(section.get("end_sec", 0.0)), 0.0), duration)
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start

        key_points = section.get("key_points") or []
        if not isinstance(key_points, list):
            key_points = [str(key_points)]

        cleaned.append(
            {
                "title": str(section.get("title", "")).strip() or "Untitled section",
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "key_points": [str(p).strip() for p in key_points if str(p).strip()],
            }
        )

    cleaned.sort(key=lambda s: s["start_sec"])
    return cleaned[:MAX_SECTIONS]


def summarize_transcript(segments: list[Any], duration: float) -> dict:
    """Summarize a transcript into timestamped thematic sections.

    Args:
        segments: Ordered Whisper segments to summarize.
        duration: Length of the source audio in seconds; section
            timestamps are clamped to this range.

    Returns:
        A dict with `sections` (see `_sanitise_sections`) and `_meta`
        carrying `model`, `input_tokens` and `output_tokens`.

    Raises:
        SummarizationUnavailable: If no Groq API key is configured.
        json.JSONDecodeError: If the model returns unparseable JSON
            despite JSON mode. The raw response is logged first so the
            failure can be diagnosed.
    """
    client = get_client()
    transcript_text = format_transcript_for_llm(segments)

    response = client.chat.completions.create(
        model=settings.summarize_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("summarize.unparseable_response", raw=raw[:2000])
        raise

    usage = response.usage
    return {
        "sections": _sanitise_sections(payload.get("sections"), duration),
        "_meta": {
            "model": settings.summarize_model,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        },
    }
