"""Preview the chunks produced from a stored transcript.

Usage:
    uv run python scripts/chunk_preview.py [transcript_id]

Loads a transcript's Whisper segments from the DB, runs chunk_segments,
prints each emitted chunk with its time span / duration / length, and
flags common anomalies (short chunks, oversized overlap, gaps, a missing
final chunk). With no id it uses the most recently created transcript.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.chunking import MIN_CHUNK_CHARS, chunk_segments
from app.db.models import Transcript
from app.db.session import SessionLocal

TARGET, OVERLAP = 45.0, 10.0


def main() -> None:
    db = SessionLocal()
    try:
        if len(sys.argv) > 1:
            tr = db.get(Transcript, sys.argv[1])
        else:
            tr = db.query(Transcript).order_by(Transcript.id).all()
            tr = tr[-1] if tr else None
        if tr is None:
            print("no transcript found in DB", file=sys.stderr)
            sys.exit(1)

        segments = tr.word_timestamps or []
        seg_end = segments[-1]["end"] if segments else 0.0
        print(f"transcript {tr.id} | lang={tr.language} | "
              f"{len(segments)} segments | audio ~{seg_end:.1f}s\n")

        chunks = list(chunk_segments(segments, TARGET, OVERLAP))
        prev = None
        for c in chunks:
            dur = c["end_sec"] - c["start_sec"]
            ov = "" if prev is None else f" overlap_prev={prev['end_sec'] - c['start_sec']:+.1f}s"
            print(f"[{c['chunk_index']:>2}] {c['start_sec']:7.1f}-{c['end_sec']:7.1f}s "
                  f"({dur:4.1f}s, {len(c['text']):>4} chars){ov}")
            print(f"      {c['text'][:110]}{'...' if len(c['text']) > 110 else ''}")
            prev = c

        # --- anomaly scan ---
        print("\n=== anomaly scan ===")
        if not chunks:
            print("!! no chunks emitted")
            return
        # Whole segments are kept in the overlap window, so the effective
        # overlap can legitimately reach OVERLAP + one segment length; only
        # flag overlaps clearly beyond that as anomalous.
        max_seg = max((s["end"] - s["start"] for s in segments), default=0.0)
        overlap_ceiling = OVERLAP + max_seg + 1.0
        print(f"chunks: {len(chunks)} | "
              f"dur min/max: {min(c['end_sec']-c['start_sec'] for c in chunks):.1f}/"
              f"{max(c['end_sec']-c['start_sec'] for c in chunks):.1f}s | "
              f"max segment: {max_seg:.1f}s (overlap ceiling ~{overlap_ceiling:.1f}s)")
        for c in chunks:
            if len(c["text"]) < MIN_CHUNK_CHARS:
                print(f"!! chunk {c['chunk_index']} shorter than {MIN_CHUNK_CHARS} chars (should be filtered)")
        for a, b in zip(chunks, chunks[1:], strict=False):
            gap = b["start_sec"] - a["end_sec"]
            if gap > 0:
                print(f"!! gap {gap:.1f}s between chunk {a['chunk_index']} and {b['chunk_index']} (no overlap)")
            elif -gap > overlap_ceiling:
                print(f"!! overlap {-gap:.1f}s between chunk {a['chunk_index']} and {b['chunk_index']} exceeds ceiling")
        covered = chunks[-1]["end_sec"]
        if seg_end - covered > 1.0:
            print(f"!! last {seg_end - covered:.1f}s of audio not covered by any chunk (missing final chunk?)")
        print("scan complete")
    finally:
        db.close()


if __name__ == "__main__":
    main()
