import json
from pathlib import Path

from agent_evals.dataset import DEFAULT_DATASET_PATH
from agent_evals.llm import StubClient
from agent_evals.runner import run_eval
from agent_evals.tracing import read_trace


def test_run_eval_offline_end_to_end(tmp_path):
    summary = run_eval(DEFAULT_DATASET_PATH, StubClient(), tmp_path, run_id="test-run")

    assert summary.total == 16
    assert summary.passed == summary.total  # deterministic stub: every sample case passes
    assert summary.pass_rate == 1.0

    written = read_trace(tmp_path / "test-run.jsonl")
    assert len(written) == summary.total
    assert all(r.run_id == "test-run" for r in written)
    assert all(r.latency_ms >= 0 for r in written)
    assert all(r.timestamp for r in written)


def test_run_eval_computes_groundedness_only_for_marked_cases(tmp_path):
    summary = run_eval(DEFAULT_DATASET_PATH, StubClient(), tmp_path, run_id="judge-run")
    marked_ids = {"kb-refund", "kb-shipping", "kb-warranty", "kb-warranty-rephrased"}
    marked = [r for r in summary.records if r.case_id in marked_ids]
    unmarked = [r for r in summary.records if r.case_id == "kb-support-hours"]

    assert len(marked) == 4
    assert all(r.groundedness_llm_score is not None for r in marked)
    assert all(r.groundedness_llm_score is None for r in unmarked)


def _write_dataset(path: Path) -> None:
    cases = [
        {
            "id": "will-pass",
            "input": "What is 2 + 2?",
            "category": "tool_use",
            "expected_facts": ["4"],
            "expected_tool_calls": ["calculator"],
            "use_llm_judge": False,
        },
        {
            "id": "will-fail",
            "input": "What is your refund policy?",
            "category": "groundedness",
            "expected_facts": ["this fact will never appear in the stub output"],
            "expected_tool_calls": ["search_kb"],
            "use_llm_judge": False,
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")


def test_pass_rate_reflects_mixed_outcomes(tmp_path):
    dataset_path = tmp_path / "mini.jsonl"
    _write_dataset(dataset_path)

    summary = run_eval(dataset_path, StubClient(), tmp_path, run_id="mini-run")

    assert summary.total == 2
    assert summary.passed == 1
    assert summary.pass_rate == 0.5


def test_threshold_gate_exits_nonzero_below_and_zero_at_or_above(tmp_path, monkeypatch, capsys):
    from agent_evals import cli as cli_module

    dataset_path = tmp_path / "mini.jsonl"
    _write_dataset(dataset_path)
    monkeypatch.setattr(cli_module, "DEFAULT_RUNS_DIR", tmp_path)

    below = cli_module.main(
        [
            "--offline",
            "run",
            "--dataset",
            str(dataset_path),
            "--run-id",
            "gate-below",
            "--threshold",
            "0.6",
        ]
    )
    capsys.readouterr()
    at_threshold = cli_module.main(
        [
            "--offline",
            "run",
            "--dataset",
            str(dataset_path),
            "--run-id",
            "gate-at",
            "--threshold",
            "0.5",
        ]
    )
    capsys.readouterr()

    assert below == 1  # pass_rate 0.5 < threshold 0.6
    assert at_threshold == 0  # pass_rate 0.5 >= threshold 0.5
