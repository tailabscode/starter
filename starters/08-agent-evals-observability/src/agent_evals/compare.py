"""`compare`: diff two runs case-by-case, flagging regressions (passed in A,
failed in B) distinctly from improvements (failed in A, passed in B)."""

from dataclasses import dataclass
from pathlib import Path

from .tracing import TraceRecord, read_trace


@dataclass(frozen=True)
class CaseDiff:
    case_id: str
    category: str
    passed_a: bool
    passed_b: bool

    @property
    def status(self) -> str:
        if self.passed_a and not self.passed_b:
            return "REGRESSION"
        if not self.passed_a and self.passed_b:
            return "IMPROVEMENT"
        return "unchanged"


@dataclass(frozen=True)
class CompareResult:
    run_id_a: str
    run_id_b: str
    diffs: list[CaseDiff]

    @property
    def regressions(self) -> list[CaseDiff]:
        return [d for d in self.diffs if d.status == "REGRESSION"]

    @property
    def improvements(self) -> list[CaseDiff]:
        return [d for d in self.diffs if d.status == "IMPROVEMENT"]


def compare_runs(records_a: list[TraceRecord], records_b: list[TraceRecord]) -> list[CaseDiff]:
    """Diff two runs, matching by `case_id`. Cases present in only one run are skipped."""
    by_id_a = {r.case_id: r for r in records_a}
    by_id_b = {r.case_id: r for r in records_b}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))
    return [
        CaseDiff(
            case_id=case_id,
            category=by_id_a[case_id].category,
            passed_a=by_id_a[case_id].passed,
            passed_b=by_id_b[case_id].passed,
        )
        for case_id in shared_ids
    ]


def load_and_compare(runs_dir: Path, run_id_a: str, run_id_b: str) -> CompareResult:
    records_a = read_trace(runs_dir / f"{run_id_a}.jsonl")
    records_b = read_trace(runs_dir / f"{run_id_b}.jsonl")
    return CompareResult(
        run_id_a=run_id_a, run_id_b=run_id_b, diffs=compare_runs(records_a, records_b)
    )
