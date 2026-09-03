"""Tests for the two upload caps on POST /transcribe.

The quota in `app.core.rate_limit` is measured in audio minutes, so it can
only run once the file has landed and ffprobe has read it. These caps are
the other half: one bounds the bytes while they are still arriving, the
other bounds a single recording. Without the first, one request fills the
disk; without the second, one visitor occupies the only worker for minutes.

Nothing here needs Postgres, Redis or a model: every path under test
rejects the request before the first database session is opened.
"""

import io

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from app.api.transcribe import _enforce_duration_limit, _spool_upload
from app.config import settings
from app.core.auth import get_api_key
from app.db.models import ApiKey
from app.main import app

ONE_MB = 1024 * 1024


@pytest.fixture
def anyio_backend():
    return "asyncio"


def upload(size_bytes: int) -> UploadFile:
    return UploadFile(file=io.BytesIO(b"\0" * size_bytes), filename="a.wav")


@pytest.mark.anyio
async def test_an_upload_over_the_size_cap_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    with pytest.raises(HTTPException) as exc:
        await _spool_upload(upload(2 * ONE_MB), str(tmp_path / "a.wav"))
    assert exc.value.status_code == 413


@pytest.mark.anyio
async def test_an_upload_at_the_size_cap_is_accepted(tmp_path, monkeypatch):
    """The cap is a maximum, not a threshold to stay under."""
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    target = tmp_path / "a.wav"
    await _spool_upload(upload(ONE_MB), str(target))
    assert target.stat().st_size == ONE_MB


@pytest.mark.anyio
async def test_the_size_cap_is_disabled_at_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    target = tmp_path / "a.wav"
    await _spool_upload(upload(2 * ONE_MB), str(target))
    assert target.stat().st_size == 2 * ONE_MB


def test_a_recording_over_the_duration_cap_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "max_audio_seconds", 90)
    with pytest.raises(HTTPException) as exc:
        _enforce_duration_limit(120.0)
    assert exc.value.status_code == 413


def test_a_recording_at_the_duration_cap_is_accepted(monkeypatch):
    monkeypatch.setattr(settings, "max_audio_seconds", 90)
    _enforce_duration_limit(90.0)


def test_the_duration_cap_is_disabled_at_zero(monkeypatch):
    monkeypatch.setattr(settings, "max_audio_seconds", 0)
    _enforce_duration_limit(7200.0)


@pytest.fixture
def authenticated_client():
    """A client whose requests carry a valid key, without a database.

    Both caps reject before `SessionLocal()` is reached, so the only thing
    standing between the test and the code under test is authentication.
    """
    app.dependency_overrides[get_api_key] = lambda: ApiKey(
        key_hash="hash-a", name="test", daily_minute_quota=60
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_the_route_deletes_the_partial_file_when_the_size_cap_trips(
    authenticated_client, tmp_path, monkeypatch
):
    """A refused upload must not leave its bytes behind.

    The spool is the shared filesystem the worker reads from; a rejection
    that leaves a file there turns every refused request into a permanent
    cost, which is the failure the cap exists to prevent.
    """
    monkeypatch.setattr(settings, "temp_audio_dir", str(tmp_path))
    monkeypatch.setattr(settings, "max_upload_mb", 1)

    response = authenticated_client.post(
        "/transcribe", files={"audio": ("a.wav", b"\0" * (2 * ONE_MB), "audio/wav")}
    )

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_the_route_deletes_the_file_when_the_recording_is_too_long(
    authenticated_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "temp_audio_dir", str(tmp_path))
    monkeypatch.setattr(settings, "max_upload_mb", 0)
    monkeypatch.setattr(settings, "max_audio_seconds", 90)
    # The file is bytes, not audio; ffprobe would reject it long before the
    # duration check, so the measurement it would produce is stubbed.
    monkeypatch.setattr("app.api.transcribe.probe_duration", lambda path: 120.0)

    response = authenticated_client.post(
        "/transcribe", files={"audio": ("a.wav", b"\0" * 1024, "audio/wav")}
    )

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []
