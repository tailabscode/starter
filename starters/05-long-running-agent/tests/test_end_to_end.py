"""Offline end-to-end run of a full job, plus a couple of CLI-surface checks."""

from __future__ import annotations

import json

from long_running_agent import cli
from long_running_agent.llm import StubClient
from long_running_agent.models import JobStatus
from long_running_agent.pipeline import DEFAULT_SOURCES
from long_running_agent.runner import execute_job
from long_running_agent.store import JobStore


def test_full_job_completes_offline_and_produces_a_real_result(tmp_path) -> None:
    with JobStore(str(tmp_path / "jobs.db")) as store:
        job = store.create_job(DEFAULT_SOURCES, max_sources=25)
        finished = execute_job(
            store,
            job.id,
            StubClient(),
            max_retries=2,
            base_delay=0.001,
            max_wall_clock_seconds=30,
            sleep_fn=lambda _: None,
        )

        assert finished.status is JobStatus.COMPLETED
        assert finished.current_step == finished.total_steps == len(DEFAULT_SOURCES) + 1

        checkpoints = store.get_checkpoints(job.id)
        assert len(checkpoints) == len(DEFAULT_SOURCES) + 1
        assert checkpoints[-1].step_name == "synthesize"

        result = json.loads(finished.result)
        assert result["source_count"] == len(DEFAULT_SOURCES)
        assert "source-alpha" in result["synthesis"]  # stub synthesis names its inputs
        # honestly labelled, never mistaken for a real model's output
        assert result["synthesis"].startswith("[stub]")


def test_cli_submit_run_offline_end_to_end(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LONG_RUNNING_AGENT_DB", str(tmp_path / "jobs.db"))
    exit_code = cli.main(["submit", "--run", "--offline"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "status: COMPLETED" in out
    assert "job_id=" in out


def test_cli_list_and_status_reflect_a_submitted_job(capsys, tmp_path, monkeypatch) -> None:
    db_path = str(tmp_path / "jobs.db")
    monkeypatch.setenv("LONG_RUNNING_AGENT_DB", db_path)

    cli.main(["submit", "--run", "--offline"])
    job_id = capsys.readouterr().out.splitlines()[0].removeprefix("job_id=")

    cli.main(["list"])
    list_out = capsys.readouterr().out
    assert job_id in list_out
    assert "COMPLETED" in list_out

    cli.main(["status", job_id])
    status_out = capsys.readouterr().out
    assert "status: COMPLETED" in status_out
    assert "5/5" in status_out  # 4 default sources + 1 synthesis step
