from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    whisper_model: str = 'small.en'
    temp_audio_dir: str = '/tmp/asr-pipeline'

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()