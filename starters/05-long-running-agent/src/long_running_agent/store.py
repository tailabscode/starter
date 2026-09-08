"""Persistent job state in SQLite. Stdlib `sqlite3` only, no ORM.

Every write that must survive a crash happens inside a single sqlite3
transaction committed in one call -- e.g. `checkpoint_step` inserts the
checkpoint row *and* advances the job row in one `commit()`, so a process
that dies partway through never leaves a checkpoint without its matching
job-row update, or vice versa.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime

from .errors import JobNotFoundError, TooManySourcesError
from .models import Checkpoint, Job, JobStatus, validate_transition

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    input TEXT NOT NULL,
    current_step INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL,
    result TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    step_index INTEGER NOT NULL,
    step_name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (job_id, step_index)
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        status=JobStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        input=row["input"],
        current_step=row["current_step"],
        total_steps=row["total_steps"],
        result=row["result"],
        error=row["error"],
    )


def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
    return Checkpoint(
        id=row["id"],
        job_id=row["job_id"],
        step_index=row["step_index"],
        step_name=row["step_name"],
        payload_json=row["payload_json"],
        created_at=row["created_at"],
    )


class JobStore:
    """Thin wrapper around a single SQLite connection holding jobs + checkpoints."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> JobStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- creation -----------------------------------------------------------

    def create_job(self, sources: list[dict], *, max_sources: int) -> Job:
        if not sources:
            raise ValueError("at least one source is required")
        if len(sources) > max_sources:
            raise TooManySourcesError(
                f"{len(sources)} sources exceeds MAX_SOURCES={max_sources}; "
                "submit fewer sources or raise the bound"
            )
        job_id = uuid.uuid4().hex[:12]
        now = _now_iso()
        total_steps = len(sources) + 1  # one step per source, plus the final synthesis step
        self._conn.execute(
            "INSERT INTO jobs "
            "(id, status, created_at, updated_at, input, current_step, total_steps, result, error) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, NULL, NULL)",
            (job_id, JobStatus.PENDING.value, now, now, json.dumps(sources), total_steps),
        )
        self._conn.commit()
        return self._require_job(job_id)

    # -- reads ----------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def _require_job(self, job_id: str) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFoundError(f"no job with id {job_id!r}")
        return job

    def list_jobs(self) -> list[Job]:
        rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at ASC").fetchall()
        return [_row_to_job(row) for row in rows]

    def get_checkpoints(self, job_id: str) -> list[Checkpoint]:
        rows = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY step_index ASC", (job_id,)
        ).fetchall()
        return [_row_to_checkpoint(row) for row in rows]

    def claim_next_pending(self) -> Job | None:
        """Return the oldest job that is ready to (re)run, or None.

        A single-process polling runner: "claiming" is just picking the oldest
        eligible job. There is no distributed lease/lock here -- documented as
        an extension idea for a multi-worker deployment.
        """
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE status IN (?, ?) ORDER BY created_at ASC LIMIT 1",
            (JobStatus.PENDING.value, JobStatus.CHECKPOINTED.value),
        ).fetchone()
        return _row_to_job(row) if row else None

    # -- transitions ------------------------------------------------------

    def mark_running(self, job_id: str) -> Job:
        """PENDING/CHECKPOINTED -> RUNNING. Idempotent if already RUNNING.

        The idempotent case handles resuming a job whose process crashed
        mid-step: the DB never got a chance to leave RUNNING, so there is
        nothing to transition -- just continue from `current_step`.
        """
        job = self._require_job(job_id)
        if job.status is JobStatus.RUNNING:
            return job
        validate_transition(job.status, JobStatus.RUNNING)
        now = _now_iso()
        self._conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            (JobStatus.RUNNING.value, now, job_id),
        )
        self._conn.commit()
        return self._require_job(job_id)

    def checkpoint_step(self, job_id: str, step_index: int, step_name: str, payload: dict) -> Job:
        """Persist a completed step's checkpoint and advance current_step, atomically."""
        job = self._require_job(job_id)
        validate_transition(job.status, JobStatus.CHECKPOINTED)
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO checkpoints (job_id, step_index, step_name, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, step_index, step_name, json.dumps(payload), now),
        )
        self._conn.execute(
            "UPDATE jobs SET status = ?, current_step = ?, updated_at = ? WHERE id = ?",
            (JobStatus.CHECKPOINTED.value, step_index + 1, now, job_id),
        )
        self._conn.commit()
        return self._require_job(job_id)

    def complete_job(self, job_id: str, result: dict) -> Job:
        job = self._require_job(job_id)
        validate_transition(job.status, JobStatus.COMPLETED)
        now = _now_iso()
        self._conn.execute(
            "UPDATE jobs SET status = ?, result = ?, updated_at = ? WHERE id = ?",
            (JobStatus.COMPLETED.value, json.dumps(result), now, job_id),
        )
        self._conn.commit()
        return self._require_job(job_id)

    def fail_job(self, job_id: str, error_message: str) -> Job:
        job = self._require_job(job_id)
        validate_transition(job.status, JobStatus.FAILED)
        now = _now_iso()
        self._conn.execute(
            "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (JobStatus.FAILED.value, error_message, now, job_id),
        )
        self._conn.commit()
        return self._require_job(job_id)

    def cancel_job(self, job_id: str) -> Job:
        job = self._require_job(job_id)
        validate_transition(job.status, JobStatus.CANCELLED)
        now = _now_iso()
        self._conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            (JobStatus.CANCELLED.value, now, job_id),
        )
        self._conn.commit()
        return self._require_job(job_id)
