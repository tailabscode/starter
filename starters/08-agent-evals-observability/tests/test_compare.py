from agent_evals.compare import compare_runs, load_and_compare
from agent_evals.tracing import TraceRecord, write_trace


def _record(case_id: str, passed: bool, category: str = "groundedness") -> TraceRecord:
    return TraceRecord(
        run_id="r",
        case_id=case_id,
        category=category,
        input="x",
        output="y",
        tool_calls=[],
        expected_facts=[],
        expected_tool_calls=[],
        keyword_coverage_score=1.0 if passed else 0.0,
        tool_use_score=1.0,
        groundedness_llm_score=None,
        passed=passed,
        latency_ms=1.0,
        timestamp="2026-01-01T00:00:00Z",
    )


def test_compare_runs_flags_regression_and_improvement():
    run_a = [_record("case-1", True), _record("case-2", False), _record("case-3", True)]
    run_b = [_record("case-1", False), _record("case-2", True), _record("case-3", True)]

    diffs = compare_runs(run_a, run_b)
    by_id = {d.case_id: d for d in diffs}

    assert by_id["case-1"].status == "REGRESSION"
    assert by_id["case-2"].status == "IMPROVEMENT"
    assert by_id["case-3"].status == "unchanged"


def test_load_and_compare_reads_from_disk(tmp_path):
    write_trace(tmp_path / "a.jsonl", [_record("case-1", True)])
    write_trace(tmp_path / "b.jsonl", [_record("case-1", False)])

    result = load_and_compare(tmp_path, "a", "b")

    assert len(result.regressions) == 1
    assert len(result.improvements) == 0
    assert result.regressions[0].case_id == "case-1"


def test_compare_only_diffs_shared_case_ids():
    run_a = [_record("only-in-a", True)]
    run_b = [_record("only-in-b", True)]
    assert compare_runs(run_a, run_b) == []


def test_load_and_compare_missing_run_raises_file_not_found(tmp_path):
    write_trace(tmp_path / "a.jsonl", [_record("case-1", True)])
    try:
        load_and_compare(tmp_path, "a", "does-not-exist")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
