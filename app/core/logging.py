"""Structured JSON logging built on structlog.

`setup_logging()` configures a structlog processor pipeline that emits
one JSON object per log line, suitable for shipping to a log
aggregator. Call it once at process startup (both the API and the RQ
worker are separate processes and each must configure its own logging).

Import `logger` anywhere to emit events:

    from app.core.logging import logger
    logger.info("job_enqueued", job_id=job_id, api_key_hash=key_hash)

Bind request- or job-scoped fields with
`structlog.contextvars.bind_contextvars(...)`; they are merged into every
subsequent event on the same context by the `merge_contextvars`
processor.
"""

import structlog


def setup_logging() -> None:
    """Configure structlog to render events as JSON.

    Pipeline: merge_contextvars -> add_log_level -> ISO TimeStamper ->
    JSONRenderer. Idempotent, safe to call more than once.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )


# Default logger to import across the codebase. structlog binds
# configuration lazily on first use, so this works even before
# setup_logging() runs; call setup_logging() at startup to pin the JSON
# pipeline.
logger = structlog.get_logger()
