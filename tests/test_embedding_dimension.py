"""EMBEDDING_DIMENSION against the size of `chunks.embedding` (audit R17): a
size the column cannot hold stops the API at startup, not at the first
upload's insert."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import EMBEDDING_COLUMN_DIMENSION, Settings
from app.models import Chunk


def test_the_default_fits_the_column():
    s = Settings(_env_file=None, database_url="postgresql://unused")

    assert s.embedding_dimension == EMBEDDING_COLUMN_DIMENSION
    assert Chunk.__table__.c.embedding.type.dim == EMBEDDING_COLUMN_DIMENSION


def test_a_size_the_column_cannot_hold_fails_at_startup():
    with pytest.raises(ValidationError, match="chunks.embedding holds 1536-dimension vectors"):
        Settings(_env_file=None, database_url="postgresql://unused", embedding_dimension=3072)
