"""Application configuration loaded from environment variables.

Settings are read from a local `.env` file (if present) and from the
process environment; unknown variables are ignored. Import the
module-level `settings` instance rather than instantiating `Settings`
directly, so values are loaded once per process.
"""

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    postgres_user: str = Field(description="Postgres role used by the application.")
    postgres_password: str = Field(description="Password for `postgres_user`.")
    postgres_db: str = Field(description="Database name on the Postgres instance.")
    postgres_host: str = Field(
        default="localhost",
        description="Postgres hostname.",
    )
    postgres_port: int = Field(
        default=5432,
        description="Postgres TCP port.",
    )

    @computed_field(description="SQLAlchemy URL built from the postgres_* fields; uses the psycopg (v3) driver.")
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
