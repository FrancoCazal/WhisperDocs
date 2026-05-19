"""Cursor-based pagination envelope.

Cursor is the `id` of the last item in the current page. Pages return items
ordered by `id DESC` (newer first). Caller passes `cursor=<last_id>` to fetch
items strictly older than that. `next_cursor` is `None` when no more pages.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    next_cursor: int | None = None
