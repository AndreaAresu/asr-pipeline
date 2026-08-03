"""FastAPI application entrypoint.

Wires the ASR model lifecycle to the application lifespan, registers
the transcription and jobs routers, and exposes a `/health` liveness
probe.
"""

import uuid

import structlog
from fastapi import FastAPI, Request

from app.api.transcribe import router as transcribe_router
from app.api.jobs import router as jobs_router
from app.api.search import router as search_router
from app.api.summarize import router as summarize_router
from app.config import settings
from app.core.logging import setup_logging, logger

setup_logging()

app = FastAPI(title="ASR Pipeline", version="0.1.0")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Bind per-request context and log request start/end.

    Reads (or mints) an `X-Request-Id`, binds it along with the path and
    method so every log event emitted while handling the request carries
    them, and echoes the id back on the response so clients can quote it
    in bug reports.
    """
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=request.url.path,
        method=request.method,
    )

    logger.info("request.start")
    response = await call_next(request)
    logger.info("request.end", status_code=response.status_code)

    response.headers["X-Request-Id"] = request_id
    return response


app.include_router(jobs_router)
app.include_router(transcribe_router)
app.include_router(search_router)
app.include_router(summarize_router)


@app.get("/health")
async def health():
    """Liveness probe returning a static OK payload."""
    return {"status": "ok"}
