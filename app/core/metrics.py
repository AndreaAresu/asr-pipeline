"""Prometheus metrics for the `/metrics` endpoint.

The interesting numbers here — how many jobs are in each state, how much
audio has been transcribed — are produced by the *worker*, in a different
process (and under compose, a different container) from the API that
serves `/metrics`. Ordinary in-process counters would therefore always
read zero: the API never runs a job, so it never increments them.

So job metrics are collected from Postgres at scrape time by
`JobStateCollector`. Postgres is already the single source of truth for
job state, which makes these numbers correct by construction: they
survive restarts, they cannot drift from reality, and they are the same
whether one API instance is running or five. The cost is a couple of
aggregate queries per scrape, which is cheap at Prometheus intervals.

HTTP-level counters stay in-process, since they describe this instance's
own request handling and no other process can observe them.
"""

from prometheus_client import CollectorRegistry, Counter
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from sqlalchemy import func

from app.core.logging import logger
from app.db.models import Job
from app.db.session import SessionLocal

# A dedicated registry rather than the global default: it keeps the
# process-wide Python GC and platform collectors out of the output, so
# what /metrics exposes is exactly what this module defines.
registry = CollectorRegistry()

http_requests_total = Counter(
    "asr_http_requests_total",
    "HTTP requests handled by this instance.",
    ["method", "path", "status"],
    registry=registry,
)

http_errors_total = Counter(
    "asr_http_errors_total",
    "HTTP responses with a 4xx or 5xx status, by status class.",
    ["status_class"],
    registry=registry,
)


class JobStateCollector:
    """Collects job metrics from Postgres on each scrape.

    Yields:
        `asr_jobs_total` — jobs by status, and
        `asr_audio_seconds_processed_total` — audio seconds successfully
        transcribed.

    A database error during a scrape is logged and swallowed: metrics
    going missing for one interval is a much better failure than the
    metrics endpoint itself returning 500 and taking down monitoring
    along with the database.
    """

    def collect(self):
        jobs = GaugeMetricFamily(
            "asr_jobs_total",
            "Transcription jobs recorded, by status.",
            labels=["status"],
        )
        audio = CounterMetricFamily(
            "asr_audio_seconds_processed",
            "Total seconds of audio transcribed successfully.",
        )

        db = SessionLocal()
        try:
            rows = db.query(Job.status, func.count(Job.id)).group_by(Job.status).all()
            # Always emit every state, so a status that has not occurred yet
            # reads 0 instead of vanishing from the output — a series that
            # disappears is much harder to alert on than one that is zero.
            counts = {"queued": 0, "processing": 0, "done": 0, "failed": 0}
            counts.update({status: count for status, count in rows})
            for status, count in counts.items():
                jobs.add_metric([status], count)

            seconds = db.query(func.coalesce(func.sum(Job.duration), 0.0)).filter(
                Job.status == "done"
            ).scalar()
            audio.add_metric([], float(seconds))
        except Exception as e:  # pragma: no cover - defensive
            logger.error("metrics.collect_failed", error=str(e))
            return
        finally:
            db.close()

        yield jobs
        yield audio


registry.register(JobStateCollector())
