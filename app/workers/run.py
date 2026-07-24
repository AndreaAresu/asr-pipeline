"""Entrypoint for the transcription RQ worker.

Run with:

    uv run python -m app.workers.run

On startup it reaps jobs orphaned in `processing` by a previously killed
worker, then starts an RQ worker on the `transcribe` queue.

Worker class by platform: on macOS the default forking worker aborts
when a job loads faster-whisper/ctranslate2, because those libraries
initialize Objective-C/Accelerate state that is unsafe to use after
`fork()` ("+[NSNumber initialize] may have been in progress ... Crashing
instead"). A non-forking `SimpleWorker` runs the job in the worker
process itself and sidesteps the crash. Linux (e.g. the Docker
deployment) keeps the standard forking `Worker`.
"""

import sys

from redis import Redis
from rq import Queue, SimpleWorker, Worker

from app.config import settings
from app.core.logging import setup_logging, logger
from app.workers.transcribe_worker import reap_stale_jobs


def main() -> None:
    setup_logging()

    reaped = reap_stale_jobs()
    if reaped:
        logger.info("worker.reaped_stale_jobs", count=reaped)

    redis = Redis.from_url(settings.redis_url)
    queue = Queue("transcribe", connection=redis)

    worker_cls = SimpleWorker if sys.platform == "darwin" else Worker
    logger.info("worker.starting", worker_class=worker_cls.__name__, queue="transcribe")

    worker = worker_cls([queue], connection=redis)
    worker.work()


if __name__ == "__main__":
    main()
