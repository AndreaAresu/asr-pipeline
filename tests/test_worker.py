"""Tests for worker-side failure accounting.

These cover the logic that decides *when* a job is dead and *why*, which
is the part that silently rots when it is wrong: a job wrongly declared
dead reports a failure that never happened, and a dead job left alone
makes a client poll forever.
"""

from datetime import timedelta

from app.workers.transcribe_worker import (
    MIN_STALE_TIMEOUT,
    killing_signal,
    stale_cutoff_for,
    termination_reason,
)


def test_a_short_job_gets_the_floor_budget():
    assert stale_cutoff_for(30) > MIN_STALE_TIMEOUT


def test_the_budget_grows_with_the_recording():
    """A 4-hour job is legitimately 'processing' for hours.

    A fixed cutoff would reap it mid-transcription and record a failure
    that never happened.
    """
    assert stale_cutoff_for(14400) > timedelta(hours=4)
    assert stale_cutoff_for(14400) > stale_cutoff_for(3600) > stale_cutoff_for(60)


def test_an_unknown_duration_still_gets_a_budget():
    """duration is nullable — a job that failed before probing has none."""
    assert stale_cutoff_for(None) >= MIN_STALE_TIMEOUT


def test_the_killing_signal_is_read_from_the_wait_status():
    assert killing_signal(9) == 9    # SIGKILL, what the OOM killer sends
    assert killing_signal(15) == 15  # SIGTERM


def test_a_missing_wait_status_is_not_an_error():
    """RQ passes None on some paths.

    Doing the arithmetic unguarded raises inside the failure handler, and
    the failure it was reporting is lost with it — which is exactly how
    this went wrong the first time.
    """
    assert killing_signal(None) is None
    assert killing_signal("unexpected") is None


def test_the_reason_names_the_signal_when_known():
    assert "signal 9" in termination_reason(9)


def test_the_reason_stays_readable_without_a_signal():
    reason = termination_reason(None)
    assert "signal" not in reason.split("OOM")[0]
    assert reason.startswith("worker process terminated before")
