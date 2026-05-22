"""SQLAlchemy ORM models for jobs and their transcripts.

`Job` tracks the lifecycle of a single transcription request; `Transcript`
holds the resulting text and per-word timestamps for jobs that completed
successfully.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Float, DateTime, ForeignKey
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

    def __repr__(self) -> str:
        return f"Transcript(id={self.id}, job_id={self.job_id}, language={self.language})"
