"""Persistent memory: a SQLite-backed store of discrete memory records.

Survives across sessions and processes -- the counterpart to `session.py`'s
deliberately non-persistent conversation list. Stdlib `sqlite3` only, no ORM.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime

from .embedder import HashingEmbedder, cosine_similarity
from .errors import MemoryNotFoundError
from .models import MemoryRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    embedding TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        content=row["content"],
        category=row["category"],
        created_at=row["created_at"],
        last_accessed_at=row["last_accessed_at"],
        source_session_id=row["source_session_id"],
        embedding=row["embedding"],
    )


class MemoryStore:
    """SQLite-backed persistent memory, with embedding-based retrieval and dedup."""

    def __init__(self, db_path: str, embedder: HashingEmbedder) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._embedder = embedder

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- basic CRUD -----------------------------------------------------------

    def create(self, content: str, category: str, source_session_id: str) -> MemoryRecord:
        record_id = uuid.uuid4().hex[:12]
        now = _now_iso()
        embedding = json.dumps(self._embedder.embed(content))
        self._conn.execute(
            "INSERT INTO memories "
            "(id, content, category, created_at, last_accessed_at, source_session_id, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record_id, content, category, now, now, source_session_id, embedding),
        )
        self._conn.commit()
        return self._require(record_id)

    def update(self, memory_id: str, content: str, category: str) -> MemoryRecord:
        self._require(memory_id)  # raises MemoryNotFoundError if missing
        now = _now_iso()
        embedding = json.dumps(self._embedder.embed(content))
        self._conn.execute(
            "UPDATE memories SET content = ?, category = ?, last_accessed_at = ?, embedding = ? "
            "WHERE id = ?",
            (content, category, now, embedding, memory_id),
        )
        self._conn.commit()
        return self._require(memory_id)

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _row_to_record(row) if row else None

    def _require(self, memory_id: str) -> MemoryRecord:
        record = self.get(memory_id)
        if record is None:
            raise MemoryNotFoundError(f"no memory with id {memory_id!r}")
        return record

    def list_all(self) -> list[MemoryRecord]:
        rows = self._conn.execute("SELECT * FROM memories ORDER BY created_at ASC").fetchall()
        return [_row_to_record(row) for row in rows]

    def delete(self, memory_id: str) -> None:
        self._require(memory_id)  # raises MemoryNotFoundError if missing
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def touch_accessed(self, memory_id: str) -> None:
        self._conn.execute(
            "UPDATE memories SET last_accessed_at = ? WHERE id = ?", (_now_iso(), memory_id)
        )
        self._conn.commit()

    # -- retrieval --------------------------------------------------------------

    def retrieve_top_k(self, query: str, k: int) -> list[tuple[MemoryRecord, float]]:
        """Top-k memories by cosine similarity to `query`, positive scores only.

        Touches (`last_accessed_at`) every memory it returns -- retrieval is
        itself a kind of access, so a memory that keeps getting surfaced looks
        recently used, and one nobody ever asks about doesn't.
        """
        query_vec = self._embedder.embed(query)
        scored: list[tuple[MemoryRecord, float]] = []
        for record in self.list_all():
            vec = json.loads(record.embedding)
            score = cosine_similarity(query_vec, vec)
            if score > 0:
                scored.append((record, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:k]
        for record, _score in top:
            self.touch_accessed(record.id)
        return top

    # -- bounded, deduplicated writes --------------------------------------------

    def upsert_with_dedup(
        self, content: str, category: str, source_session_id: str, threshold: float
    ) -> tuple[MemoryRecord, str]:
        """Create a memory, or update an existing near-duplicate in place.

        Searches every existing memory for the highest cosine similarity to
        `content`. If it clears `threshold`, that memory is updated in place
        (same id, new content/category/embedding) instead of inserting a
        duplicate. Otherwise a new record is created. Returns
        `(record, "update" | "create")`.
        """
        new_vec = self._embedder.embed(content)
        best_match: MemoryRecord | None = None
        best_score = 0.0
        for record in self.list_all():
            existing_vec = json.loads(record.embedding)
            score = cosine_similarity(new_vec, existing_vec)
            if score > best_score:
                best_score = score
                best_match = record

        if best_match is not None and best_score >= threshold:
            updated = self.update(best_match.id, content, category)
            return updated, "update"

        created = self.create(content, category, source_session_id)
        return created, "create"
