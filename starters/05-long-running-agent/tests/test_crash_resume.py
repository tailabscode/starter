"""The most important test: a real process crash mid-job, then a real resume.

This drives the actual CLI in a subprocess so `--simulate-crash-after-step`
can call the genuine `os._exit(1)` hard-kill path (not a mocked stand-in),
proving durability comes from the SQLite commits, not from any in-memory
state or graceful-shutdown code. A second subprocess then resumes the same
job against the same database file and must pick up exactly where the first
one died -- verified by checking that the checkpoints for already-finished
steps are byte-for-byte untouched (same `created_at`), while only the
remaining steps get new checkpoint rows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from long_running_agent.models import JobStatus
from long_running_agent.store import JobStore

SOURCES = [
    {"name": "src-a", "content": "Content A mentions the number 10 and Acme Corp."},
    {"name": "src-b", "content": "Content B mentions the number 20 and Beta Inc."},
    {"name": "src-c", "content": "Content C mentions the number 30 and Gamma LLC."},
    {"name": "src-d", "content": "Content D mentions the number 40 and Delta Co."},
]


def _run_cli(args: list[str], db_path: str, timeout: float = 30) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["LONG_RUNNING_AGENT_DB"] = db_path
    return subprocess.run(
        [sys.executable, "-m", "long_running_agent", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _job_id_from_stdout(stdout: str) -> str:
    first_line = stdout.strip().splitlines()[0]
    assert first_line.startswith("job_id=")
    return first_line.removeprefix("job_id=")


def test_crash_mid_job_then_resume_skips_completed_steps(tmp_path) -> None:
    db_path = str(tmp_path / "jobs.db")

    submit_result = _run_cli(
        [
            "submit",
            "--run",
            "--offline",
            "--sources-json",
            json.dumps(SOURCES),
            "--simulate-crash-after-step",
            "2",
        ],
        db_path,
    )
    # os._exit(1) is a hard kill -- the process never gets to return 0 cleanly.
    assert submit_result.returncode == 1
    job_id = _job_id_from_stdout(submit_result.stdout)

    with JobStore(db_path) as store:
        crashed_job = store.get_job(job_id)
        assert crashed_job is not None
        assert crashed_job.status is JobStatus.CHECKPOINTED
        assert crashed_job.current_step == 3  # steps 0, 1, 2 done; step 3 and 4 remain

        checkpoints_after_crash = store.get_checkpoints(job_id)
        assert [c.step_name for c in checkpoints_after_crash] == [
            "process_source:src-a",
            "process_source:src-b",
            "process_source:src-c",
        ]
        step0_created_at_before_resume = checkpoints_after_crash[0].created_at
        step0_payload_before_resume = checkpoints_after_crash[0].payload_json

    resume_result = _run_cli(["resume", job_id, "--offline"], db_path)
    assert resume_result.returncode == 0, resume_result.stderr

    with JobStore(db_path) as store:
        finished_job = store.get_job(job_id)
        assert finished_job is not None
        assert finished_job.status is JobStatus.COMPLETED
        assert finished_job.current_step == finished_job.total_steps == 5

        checkpoints_after_resume = store.get_checkpoints(job_id)
        assert [c.step_name for c in checkpoints_after_resume] == [
            "process_source:src-a",
            "process_source:src-b",
            "process_source:src-c",
            "process_source:src-d",
            "synthesize",
        ]

        # The proof that finished steps were not redone: the exact same
        # checkpoint row for step 0 survived, untouched, across the resume.
        assert checkpoints_after_resume[0].created_at == step0_created_at_before_resume
        assert checkpoints_after_resume[0].payload_json == step0_payload_before_resume

        result = json.loads(finished_job.result)
        assert result["source_count"] == 4
        assert "synthesis" in result
