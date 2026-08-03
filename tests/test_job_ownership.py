"""Tests that a job is only readable by the key that submitted it.

Authentication alone would not be enough here: every endpoint could
require a valid key and *any* valid key would still read *any* job. What
matters is the ownership check, and that a job belonging to someone else
is indistinguishable from one that does not exist.
"""

import pytest
from fastapi import HTTPException

from app.api.jobs import _owned_job
from app.db.models import ApiKey, Job


class FakeSession:
    """Stands in for a Session, returning one preloaded job by id."""

    def __init__(self, job: Job | None):
        self._job = job

    def get(self, model, pk):
        return self._job if self._job is not None and self._job.id == pk else None


def key(key_hash: str) -> ApiKey:
    return ApiKey(key_hash=key_hash, name="test", daily_minute_quota=60)


def job(job_id: str, owner_hash: str) -> Job:
    return Job(id=job_id, api_key_hash=owner_hash, audio_filename="a.wav", status="done")


def test_the_owner_can_read_their_job():
    owned = job("job-1", "hash-a")
    assert _owned_job(FakeSession(owned), "job-1", key("hash-a")) is owned


def test_another_key_cannot_read_it():
    with pytest.raises(HTTPException) as exc:
        _owned_job(FakeSession(job("job-1", "hash-a")), "job-1", key("hash-b"))
    assert exc.value.status_code == 404


def test_a_missing_job_is_also_404():
    with pytest.raises(HTTPException) as exc:
        _owned_job(FakeSession(None), "job-1", key("hash-a"))
    assert exc.value.status_code == 404


def test_the_two_denials_are_indistinguishable():
    """404 for both, never 403.

    Answering 403 for someone else's job would confirm the id exists,
    turning the endpoint into an oracle for probing valid job ids.
    """
    with pytest.raises(HTTPException) as foreign:
        _owned_job(FakeSession(job("job-1", "hash-a")), "job-1", key("hash-b"))
    with pytest.raises(HTTPException) as missing:
        _owned_job(FakeSession(None), "job-1", key("hash-a"))

    assert foreign.value.status_code == missing.value.status_code == 404
    assert foreign.value.detail == missing.value.detail
