"""FastAPI application entrypoint.

Wires the ASR model lifecycle to the application lifespan, registers
the transcription and jobs routers, and exposes a `/health` liveness
probe.
"""

from fastapi import FastAPI

from app.api.transcribe import router as transcribe_router
from app.api.jobs import router as jobs_router
from app.config import settings

app = FastAPI(title="ASR Pipeline", version="0.1.0")

app.include_router(jobs_router)
app.include_router(transcribe_router)


@app.get("/health")
async def health():
    """Liveness probe returning a static OK payload."""
    return {"status": "ok"}
