"""Tests for database URL resolution.

The failure this guards against is a deploy-day one: managed Postgres
providers hand out `postgres://…`, SQLAlchemy 2 needs the driver named,
and the resulting error surfaces far from its cause.
"""

import pytest

from app.config import Settings


def build(**kwargs) -> Settings:
    """Construct Settings without reading the developer's local .env."""
    return Settings(_env_file=None, **kwargs)


def test_url_is_assembled_from_the_postgres_parts():
    settings = build(
        postgres_user="asr",
        postgres_password="pw",
        postgres_db="asr_pipeline",
        postgres_host="db.internal",
        postgres_port=6432,
    )
    assert settings.database_url == "postgresql+psycopg://asr:pw@db.internal:6432/asr_pipeline"


@pytest.mark.parametrize(
    "given",
    ["postgres://u:p@host.flycast:5432/db", "postgresql://u:p@host.flycast:5432/db"],
)
def test_provider_prefixes_are_rewritten_to_the_psycopg_driver(given):
    assert build(DATABASE_URL=given).database_url == "postgresql+psycopg://u:p@host.flycast:5432/db"


def test_an_explicit_driver_is_left_alone():
    url = "postgresql+psycopg://u:p@h/db"
    assert build(DATABASE_URL=url).database_url == url


def test_the_url_overrides_the_parts():
    settings = build(
        DATABASE_URL="postgresql+psycopg://override:pw@remote/db",
        postgres_user="ignored",
        postgres_password="ignored",
        postgres_db="ignored",
    )
    assert "override" in settings.database_url
    assert "ignored" not in settings.database_url


def test_an_incomplete_configuration_fails_fast():
    """Better a startup error than 'None' interpolated into a DSN."""
    with pytest.raises(ValueError, match="database is not configured"):
        build(postgres_user="asr")  # no password, no db, no URL
