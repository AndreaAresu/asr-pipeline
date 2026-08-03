"""SQLAlchemy ORM models for jobs and their transcripts.

`Job` tracks the lifecycle of a single transcription request; `Transcript`
holds the resulting text and per-word timestamps for jobs that completed
successfully.
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Float, DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


class Job(Base):
    """Lifecycle record for a single transcription request."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    api_key_hash: Mapped[str] = mapped_column(String, default="dev")
    audio_filename: Mapped[str] = mapped_column(String)
    duration: Mapped[float | None] = mapped_column(Float)
    # valid values: queued | processing | done | failed
    status: Mapped[str] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    transcript: Mapped["Transcript | None"] = relationship(back_populates="job", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"Job(id={self.id}, status={self.status}, created_at={self.created_at})"


class Transcript(Base):
    """Successful transcription output associated with a `Job`."""

    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String, ForeignKey("jobs.id"))
    full_text: Mapped[str] = mapped_column(String)
    language: Mapped[str | None] = mapped_column(String)
    word_timestamps: Mapped[dict] = mapped_column(JSONB)

    job: Mapped["Job"] = relationship(back_populates="transcript")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="transcript", cascade="all, delete-orphan")
    summary: Mapped["Summary | None"] = relationship(
        back_populates="transcript", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Transcript(id={self.id}, job_id={self.job_id}, language={self.language})"


class Chunk(Base):
    """A retrievable slice of a transcript with its embedding vector.

    Transcripts are split into overlapping-or-contiguous chunks so they can
    be semantically searched: each chunk carries its text, the time span it
    covers in the source audio, and a 384-dim embedding (the output size of
    the sentence-transformers `all-MiniLM-L6-v2` family) stored as a
    pgvector column.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    transcript_id: Mapped[str] = mapped_column(String, ForeignKey("transcripts.id"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(String)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))

    transcript: Mapped["Transcript"] = relationship(back_populates="chunks")

    # TODO: once enough chunks exist to train the clustering (well past a
    # few thousand rows), add an IVFFlat index for approximate search:
    #   Index("ix_chunks_embedding", Chunk.embedding,
    #         postgresql_using="ivfflat",
    #         postgresql_with={"lists": 100},
    #         postgresql_ops={"embedding": "vector_cosine_ops"})
    # Until then the table uses a sequential scan, which is exact and fast
    # enough up to ~10k chunks. IVFFlat needs data present at CREATE time to
    # train its clusters, so it must be built after backfilling, not here.

    def __repr__(self) -> str:
        return f"Chunk(id={self.id}, transcript_id={self.transcript_id}, chunk_index={self.chunk_index})"

class Summary(Base):
    """Cached LLM summary of a transcript.

    Summarization is the only part of the pipeline that costs money and
    seconds per call, and its input (the transcript) never changes once
    written — so the result is cached rather than recomputed. The
    transcript id doubles as the primary key, which makes "one summary
    per transcript" a constraint the database enforces rather than
    something the endpoint has to remember.

    `summary_json` holds the model's structured output verbatim: the
    thematic `sections` (each with title, start/end seconds and key
    points) plus a `_meta` block recording the model and token counts
    that produced it, kept for cost auditing.
    """

    __tablename__ = "summaries"

    transcript_id: Mapped[str] = mapped_column(String, ForeignKey("transcripts.id"), primary_key=True)
    summary_json: Mapped[dict] = mapped_column(JSONB)
    model_used: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    transcript: Mapped["Transcript"] = relationship(back_populates="summary")

    def __repr__(self) -> str:
        return f"Summary(transcript_id={self.transcript_id}, model_used={self.model_used})"


class ApiKey(Base):
    """An API key, stored only as the SHA-256 hash of its plaintext.

    `daily_minute_quota` is the key's allowance of *audio minutes* per
    rolling 24h window (see `app.core.rate_limit`), not a request count —
    a long upload costs proportionally more than a short one.
    """

    __tablename__ = "api_keys"

    key_hash: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    daily_minute_quota: Mapped[int] = mapped_column(Integer, default=60)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"ApiKey(key_hash={self.key_hash}, name={self.name})"

