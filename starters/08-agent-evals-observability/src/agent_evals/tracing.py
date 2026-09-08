"""Structured trace records: one JSONL line per eval case, written to `runs/`.

This is the starter's observability surface — real, inspectable records
(inputs, tool calls, output, every metric's score, latency, a run id, a
wall-clock timestamp), not a mock.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# src/agent_evals/tracing.py -> parents[2] is the starter root, where runs/ lives.
DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


@dataclass(frozen=True)
class ToolCallTrace:
    name: str
    input: dict[str, Any]
    output: str


@dataclass(frozen=True)
class TraceRecord:
    run_id: str
    case_id: str
    category: str
    input: str
    output: str
    tool_calls: list[ToolCallTrace]
    expected_facts: list[str]
    expected_tool_calls: list[str]
    keyword_coverage_score: float
    tool_use_score: float
    groundedness_llm_score: float | None
    passed: bool
    latency_ms: float
    timestamp: str


def new_run_id() -> str:
    """A sortable, git-independent run id: wall-clock time plus a short random suffix."""
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_trace(path: Path, records: list[TraceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def read_trace(path: Path) -> list[TraceRecord]:
    if not path.exists():
        raise FileNotFoundError(f"No run found at {path}")
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["tool_calls"] = [ToolCallTrace(**tc) for tc in data["tool_calls"]]
            records.append(TraceRecord(**data))
    return records
