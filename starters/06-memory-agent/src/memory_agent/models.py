"""The one persistent row type, and the categories a memory can belong to."""

from __future__ import annotations

from dataclasses import dataclass

MEMORY_CATEGORIES = ("preference", "fact", "episodic")


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    content: str
    category: str
    created_at: str
    last_accessed_at: str
    source_session_id: str
    embedding: str  # JSON-encoded list[float]
