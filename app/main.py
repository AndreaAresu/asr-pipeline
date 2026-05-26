"""FastAPI application entrypoint.

Wires the ASR model lifecycle to the application lifespan, registers the
transcription router, and exposes a `/health` liveness probe.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.transcribe import router as transcribe_router
from app.api.jobs import router as jobs_router
from app.config import settings
from app.core import ASRModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down process-wide resources.

    Loads the Whisper model once on startup and attaches it to
    `app.state.asr` so every request reuses the same instance instead of
    paying the load cost per call.
    """
    app.state.asr = ASRModel(settings.whisper_model)
    print("startup")

    yield

    print("shutdown")


app = FastAPI(lifespan=lifespan, title="ASR Pipeline", version="0.1.0")

app.include_router(jobs_router)
app.include_router(transcribe_router)


@app.get("/health")
async def health():
    """Liveness probe returning a static OK payload."""
    return {"status": "ok"}
