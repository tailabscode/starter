"""ResearchState round-trips through JSON, and the runs-dir helpers behave."""

from __future__ import annotations

import pytest

from multi_agent_research.errors import RunNotFoundError
from multi_agent_research.state import ResearchState


def test_state_round_trips_through_json(tmp_path) -> None:
    state = ResearchState(run_id="abc123", question="What drives cost?")
    state.plan = {"reasoning": "split by topic", "subtasks": []}
    state.subtasks = [{"id": "mining", "description": "Research mining"}]
    state.findings = [{"subtask_id": "mining", "answer": "y", "sources": []}]
    state.critiques = [{"is_sufficient": True, "gaps": [], "reasoning": "z"}]
    state.revision_round = 1
    state.termination_reason = "sufficient"
    state.final_report = "final report text"
    state.add_usage({"input_tokens": 10, "output_tokens": 5})

    path = state.save(tmp_path)
    assert path.exists()

    loaded = ResearchState.load(tmp_path, "abc123")

    assert loaded == state
    assert loaded.token_usage == {"input_tokens": 10, "output_tokens": 5}
    assert loaded.final_report == "final report text"


def test_list_runs_returns_sorted_run_ids(tmp_path) -> None:
    ResearchState(run_id="r2", question="q2").save(tmp_path)
    ResearchState(run_id="r1", question="q1").save(tmp_path)

    assert ResearchState.list_runs(tmp_path) == ["r1", "r2"]


def test_list_runs_empty_dir_returns_empty_list(tmp_path) -> None:
    assert ResearchState.list_runs(tmp_path / "does-not-exist-yet") == []


def test_load_missing_run_raises_named_error(tmp_path) -> None:
    with pytest.raises(RunNotFoundError):
        ResearchState.load(tmp_path, "does-not-exist")
