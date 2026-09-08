"""The `run` command: execute every eval case against the example agent,
score it with all applicable metrics, write a trace record, and report the
aggregate pass rate for the threshold gate.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from .dataset import EvalCase, load_cases
from .example_agent import run_example_agent
from .llm import LLMClient
from .scoring import score_keyword_coverage, score_tool_use
from .tracing import ToolCallTrace, TraceRecord, new_run_id, now_iso, write_trace

DEFAULT_THRESHOLD = 0.8
# Pass bar for the (optional) LLM-judge groundedness score; keyword coverage
# and tool-use correctness must always be perfect for a case to pass.
GROUNDEDNESS_PASS_THRESHOLD = 0.5


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    total: int
    passed: int
    pass_rate: float
    records: list[TraceRecord]


def run_eval(
    dataset_path: Path,
    client: LLMClient,
    runs_dir: Path,
    run_id: str | None = None,
) -> RunSummary:
    """Run every case in `dataset_path` against the example agent, write
    `runs_dir/<run_id>.jsonl`, and return the aggregate summary."""
    cases = load_cases(dataset_path)
    run_id = run_id or new_run_id()
    records = [_run_one_case(client, run_id, case) for case in cases]
    write_trace(runs_dir / f"{run_id}.jsonl", records)
    passed = sum(1 for r in records if r.passed)
    total = len(records)
    return RunSummary(
        run_id=run_id,
        total=total,
        passed=passed,
        pass_rate=(passed / total if total else 0.0),
        records=records,
    )


def _run_one_case(client: LLMClient, run_id: str, case: EvalCase) -> TraceRecord:
    start = time.monotonic()
    output = run_example_agent(client, case.input)
    latency_ms = (time.monotonic() - start) * 1000

    keyword_score = score_keyword_coverage(output.text, case.expected_facts)
    tool_score = score_tool_use(output.tool_calls, case.expected_tool_calls)

    groundedness_score = None
    if case.use_llm_judge:
        judged = client.judge_groundedness(case.input, output.text, case.expected_facts)
        groundedness_score = float(judged["score"])

    passed = (
        keyword_score == 1.0
        and tool_score == 1.0
        and (groundedness_score is None or groundedness_score >= GROUNDEDNESS_PASS_THRESHOLD)
    )

    return TraceRecord(
        run_id=run_id,
        case_id=case.id,
        category=case.category,
        input=case.input,
        output=output.text,
        tool_calls=[
            ToolCallTrace(name=c.name, input=c.input, output=c.output) for c in output.tool_calls
        ],
        expected_facts=case.expected_facts,
        expected_tool_calls=case.expected_tool_calls,
        keyword_coverage_score=keyword_score,
        tool_use_score=tool_score,
        groundedness_llm_score=groundedness_score,
        passed=passed,
        latency_ms=round(latency_ms, 3),
        timestamp=now_iso(),
    )
