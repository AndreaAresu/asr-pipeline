import uuid
from datetime import datetime
from sqlalchemy import String, Float, JSON, DateTime, Integer, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    api_key_hash: Mapped[str] = mapped_column(String, default='dev')
    audio_filename: Mapped[str] = mapped_column(String)
    duration: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String) # queued|processing|done|failed
    error_message: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    def __repr__(self) -> str:
        return f"Job(id={self.id}, status={self.status}, created_at={self.created_at})"
    
class Transcript(Base):
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String, ForeignKey('jobs.id'))
    full_text: Mapped[str] = mapped_column(String)
    language: Mapped[str | None] = mapped_column(String)
    word_timestamps: Mapped[dict] = mapped_column(JSON)

