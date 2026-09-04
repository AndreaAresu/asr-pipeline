"""FastAPI application entrypoint.

Registers the request-context middleware, the five routers and the MCP
route, and serves the three unauthenticated routes: the browser console at
`/`, the `/health` liveness probe, and `/metrics`. No model is loaded
here, the API only enqueues work; transcription happens in the worker
process.
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.routing import Route

from app.api.jobs import router as jobs_router
from app.api.search import router as search_router
from app.api.summarize import router as summarize_router
from app.api.transcribe import router as transcribe_router
from app.api.transcripts import router as transcripts_router
from app.core.logging import logger, setup_logging
from app.core.metrics import http_errors_total, http_requests_total, registry
from app.mcp.asgi import MCP_PATH, ApiKeyGate
from app.mcp.server import mcp, mcp_asgi_app

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run the MCP session manager for as long as the app is up.

    `streamable_http_app()` wires this into the lifespan of the Starlette
    it returns, but that app is never started by anything: the route below
    only borrows it as an endpoint. Without this the route answers, then
    the first MCP message dies on "Task group is not initialized".
    """
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="ASR Pipeline", version="0.1.0", lifespan=lifespan)


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
app.include_router(transcripts_router)

# An exact Starlette route, deliberately not `app.mount()`: mounting on
# "/mcp" answers 307 to "/mcp/", which a client that does not follow
# redirects on POST never recovers from, and mounting on "/" turns every
# 404 in the API from JSON into text/plain. Both were measured; see
# `.scratch/mcp-server/research/02-libreria-e-montaggio.md`.
# GET and DELETE are inert under stateless HTTP but are declared so a
# client that tries them reads an MCP error rather than a bare 405.
app.router.routes.append(
    Route(MCP_PATH, endpoint=ApiKeyGate(mcp_asgi_app), methods=["GET", "POST", "DELETE"])
)


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
