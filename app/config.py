"""Application configuration loaded from environment variables.

Settings are read from a local `.env` file (if present) and from the
process environment; unknown variables are ignored. Import the
module-level `settings` instance rather than instantiating `Settings`
directly, so values are loaded once per process.
"""

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Managed Postgres providers hand out URLs with these prefixes, but
# SQLAlchemy 2 needs the driver spelled out to pick psycopg v3.
_DRIVER_PREFIXES = ("postgresql+psycopg://", "postgresql+psycopg2://", "postgresql+asyncpg://")


class Settings(BaseSettings):
    """Runtime configuration for the ASR pipeline."""

    whisper_model: str = Field(
        default="small.en",
        description="Whisper model identifier passed to faster-whisper (e.g. 'small.en', 'medium', 'large-v3').",
    )
    temp_audio_dir: str = Field(
        default="/tmp/asr-pipeline",
        description="Directory where uploaded audio files are buffered during transcription.",
    )
    max_upload_mb: int = Field(
        default=500,
        description=(
            "Largest upload accepted by POST /transcribe, in megabytes; 0 disables the check. "
            "Enforced while the bytes are being written, not afterwards, because the duration "
            "check that follows can only run once the whole file has landed. Lower it sharply "
            "on a public deployment: without it one request can fill the disk."
        ),
    )
    max_audio_seconds: int = Field(
        default=0,
        description=(
            "Longest recording accepted by POST /transcribe, in seconds; 0 disables the check. "
            "Off by default so local indexing of long recordings works, and set low on a public "
            "deployment, where transcription is minutes of CPU a visitor waits through."
        ),
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "sentence-transformers model used to embed chunks and search queries. "
            "Must output 384 dimensions to match the chunks.embedding column."
        ),
    )
    database_url_env: str | None = Field(
        default=None,
        validation_alias="DATABASE_URL",
        description=(
            "Full connection URL, overriding the postgres_* fields. Set by managed "
            "providers (Fly, Upstash, Neon) that hand out one string rather than parts. "
            "A bare postgres:// or postgresql:// prefix is rewritten to postgresql+psycopg://."
        ),
    )
    postgres_user: str | None = Field(default=None, description="Postgres role used by the application.")
    postgres_password: str | None = Field(default=None, description="Password for `postgres_user`.")
    postgres_db: str | None = Field(default=None, description="Database name on the Postgres instance.")
    postgres_host: str = Field(
        default="localhost",
        description="Postgres hostname.",
    )
    postgres_port: int = Field(
        default=5432,
        description="Postgres TCP port.",
    )

    @model_validator(mode="after")
    def _require_a_database(self) -> "Settings":
        """Fail fast when neither way of specifying the database is complete.

        Without this the missing parts would be interpolated as the string
        "None" and surface much later as an unintelligible connection error.
        """
        if self.database_url_env:
            return self
        missing = [
            name
            for name in ("postgres_user", "postgres_password", "postgres_db")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(
                "database is not configured: set DATABASE_URL, or all of "
                + ", ".join(n.upper() for n in missing)
            )
        return self

    @computed_field(
        description="SQLAlchemy URL: DATABASE_URL when set, otherwise built from the postgres_* fields."
    )
    @property
    def database_url(self) -> str:
        if self.database_url_env:
            url = self.database_url_env
            if url.startswith(_DRIVER_PREFIXES):
                return url
            # Normalise the provider-style prefixes to the psycopg v3 driver.
            _, _, rest = url.partition("://")
            return f"postgresql+psycopg://{rest}"
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL used by the RQ task queue.",
    )
    groq_api_key: str = Field(
        default="",
        description=(
            "Groq API key used for summarization. Optional: transcription and search "
            "work without it, and /summarize returns 503 while it is unset."
        ),
    )
    summarize_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq chat model used to summarize transcripts.",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
