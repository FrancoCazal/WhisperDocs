"""Secrets out of logs (audit A4) and explicit CORS origins (audit R7)."""

from __future__ import annotations

from app.agent.checkpointer import _redact_dsn
from app.config import Settings


def test_dsn_password_is_redacted():
    dsn = "postgresql://whisperdocs:s3cret@db:5432/whisperdocs"

    redacted = _redact_dsn(dsn)

    assert "s3cret" not in redacted
    assert "whisperdocs:" not in redacted
    assert redacted == "postgresql://***@db:5432/whisperdocs"


def test_dsn_without_credentials_is_unchanged():
    assert _redact_dsn("postgresql://db:5432/x") == "postgresql://db:5432/x"


def test_cors_default_is_not_a_wildcard():
    origins = Settings(_env_file=None, database_url="postgresql://unused").cors_allowed_origins

    assert "*" not in origins
    assert "http://localhost:8080" in origins
