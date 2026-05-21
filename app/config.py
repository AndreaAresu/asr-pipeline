"""Application configuration loaded from environment variables.

Settings are read from a local `.env` file (if present) and from the
process environment; unknown variables are ignored. Import the
module-level `settings` instance rather than instantiating `Settings`
directly, so values are loaded once per process.
"""

from pydantic import Field
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
