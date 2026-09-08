"""Explicit, JSON-serializable run state.

The whole point of writing this out as a dataclass (rather than passing loose variables between
functions) is that a run is fully inspectable and resumable-by-inspection: every stage of
``orchestrator.run_research`` saves the state to disk immediately after it finishes, so
``show <run-id>`` can display exactly how far a run got even if a later stage fails or the
process is killed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agent_research.errors import RunNotFoundError


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ResearchState:
    """The full record of one research run."""

    run_id: str
    question: str
    plan: dict[str, Any] = field(default_factory=dict)
    subtasks: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    critiques: list[dict[str, Any]] = field(default_factory=list)
    revision_round: int = 0
    termination_reason: str | None = None
    final_report: str | None = None
    token_usage: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}
    )
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def add_usage(self, usage: dict[str, int]) -> None:
        self.token_usage["input_tokens"] += usage.get("input_tokens", 0)
        self.token_usage["output_tokens"] += usage.get("output_tokens", 0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResearchState:
        return cls(**payload)

    @classmethod
    def from_json(cls, raw: str) -> ResearchState:
        return cls.from_dict(json.loads(raw))

    def save(self, runs_dir: Path) -> Path:
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        path = runs_dir / f"{self.run_id}.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, runs_dir: Path, run_id: str) -> ResearchState:
        path = runs_dir / f"{run_id}.json"
        if not path.exists():
            raise RunNotFoundError(f"No run found with id {run_id!r} in {runs_dir}")
        return cls.from_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def list_runs(runs_dir: Path) -> list[str]:
        if not runs_dir.exists():
            return []
        return sorted(p.stem for p in runs_dir.glob("*.json"))
