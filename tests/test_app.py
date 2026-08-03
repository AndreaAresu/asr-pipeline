"""Application-level smoke tests.

Deliberately narrow: they assert the app imports, the routes are wired,
and unauthenticated callers are turned away. Anything that needs Postgres,
Redis or a model lives in `scripts/smoke_test.sh`, which runs the real
stack — faking those here would test the fakes.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_every_endpoint_is_registered(client):
    paths = {route.path for route in app.routes}
    assert {
        "/health",
        "/metrics",
        "/transcribe",
        "/jobs/{job_id}",
        "/jobs/{job_id}/result",
        "/search",
        "/summarize/{transcript_id}",
    } <= paths


def test_request_id_is_echoed_back(client):
    response = client.get("/health", headers={"X-Request-Id": "abc-123"})
    assert response.headers["X-Request-Id"] == "abc-123"


def test_a_request_id_is_minted_when_absent(client):
    assert client.get("/health").headers.get("X-Request-Id")


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/search", {"json": {"query": "x"}}),
        ("post", "/summarize/some-id", {}),
    ],
)
def test_protected_endpoints_reject_missing_keys(client, method, path, kwargs):
    assert getattr(client, method)(path, **kwargs).status_code == 401


def test_metrics_are_exposed_in_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    # The HTTP counter is in-process, so it is present with no database.
    assert "asr_http_requests_total" in response.text
