"""FastAPI application entrypoint.

Wires the ASR model lifecycle to the application lifespan, registers
the transcription and jobs routers, and exposes a `/health` liveness
probe.
"""

import uuid
from pathlib import Path

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.jobs import router as jobs_router
from app.api.search import router as search_router
from app.api.summarize import router as summarize_router
from app.api.transcribe import router as transcribe_router
from app.core.logging import logger, setup_logging
from app.core.metrics import http_errors_total, http_requests_total, registry

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

    # Label on the route template ("/jobs/{job_id}"), not the raw path:
    # one series per route, rather than one per job id.
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    http_requests_total.labels(request.method, path, response.status_code).inc()
    if response.status_code >= 400:
        http_errors_total.labels(f"{response.status_code // 100}xx").inc()

    response.headers["X-Request-Id"] = request_id
    return response


app.include_router(jobs_router)
app.include_router(transcribe_router)
app.include_router(search_router)
app.include_router(summarize_router)


UI_FILE = Path(__file__).parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
async def ui():
    """Serve the browser console.

    The UI is one self-contained HTML file served by the API itself, so
    running the service is the only setup a user needs, no second
    process, no build step, and no host to configure, since the page
    calls the same origin that served it.
    """
    return FileResponse(UI_FILE)


@app.get("/health")
async def health():
    """Liveness probe returning a static OK payload."""
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Expose Prometheus metrics in the text exposition format.

    Deliberately unauthenticated so a scraper needs no credentials; it
    exposes only aggregate counts, never transcript content or key
    material. Keep it off the public internet in a real deployment.
    """
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
