"""SQLite-backed storage for todo tasks.

Why SQLite: it's stdlib (no dependency), gives us real persistence and
transactional writes, and is a realistic choice for a small local CLI tool
-- unlike a flat JSON file, concurrent CLI invocations won't corrupt data.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass(frozen=True)
class Task:
    """One todo item. Frozen because tasks are read back fresh from SQLite
    rather than mutated in place -- avoids stale in-memory state drifting
    from the database, which is the source of truth."""

    id: int
    description: str
    done: bool


class TaskStore:
    """Thin wrapper around a SQLite database of tasks.

    Opens a fresh connection per operation rather than holding one open for
    the process lifetime -- this is a short-lived CLI, not a long-running
    server, so the extra connect() cost is irrelevant and it sidesteps any
    cross-invocation locking surprises.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, description: str) -> Task:
        """Insert a new task and return it. Raises ValueError on blank input."""
        cleaned = description.strip()
        if not cleaned:
            raise ValueError("Task description cannot be empty")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO tasks (description, done) VALUES (?, 0)",
                (cleaned,),
            )
            task_id = cur.lastrowid
        assert task_id is not None  # AUTOINCREMENT guarantees this
        return Task(id=task_id, description=cleaned, done=False)

    def list(self, *, include_done: bool = True) -> list[Task]:
        """Return tasks ordered by id. Set include_done=False for pending only."""
        query = "SELECT id, description, done FROM tasks"
        if not include_done:
            query += " WHERE done = 0"
        query += " ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [Task(id=r[0], description=r[1], done=bool(r[2])) for r in rows]

    def complete(self, task_id: int) -> Task:
        """Mark a task done and return its updated state. Raises KeyError if missing."""
        with self._connect() as conn:
            cur = conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))
            if cur.rowcount == 0:
                raise KeyError(f"No task with id {task_id}")
            row = conn.execute(
                "SELECT id, description, done FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return Task(id=row[0], description=row[1], done=bool(row[2]))

    def remove(self, task_id: int) -> None:
        """Delete a task. Raises KeyError if it does not exist."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            if cur.rowcount == 0:
                raise KeyError(f"No task with id {task_id}")
