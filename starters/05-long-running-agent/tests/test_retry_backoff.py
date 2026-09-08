"""Retry with exponential backoff: recovers within budget, exhausts cleanly otherwise."""

import pytest

from long_running_agent.errors import RetryableStepError, StepFailedError
from long_running_agent.llm import StubClient
from long_running_agent.models import JobStatus
from long_running_agent.pipeline import run_step_with_retry, step_process_source
from long_running_agent.runner import execute_job
from long_running_agent.store import JobStore


def test_retry_recovers_before_exhausting_budget() -> None:
    calls: list[int] = []

    def flaky(attempt: int) -> str:
        calls.append(attempt)
        if attempt < 2:
            raise RetryableStepError(f"transient failure on attempt {attempt}")
        return "ok"

    delays: list[float] = []
    result = run_step_with_retry(flaky, max_retries=3, base_delay=0.01, sleep_fn=delays.append)

    assert result == "ok"
    assert calls == [0, 1, 2]  # failed twice, succeeded on the third attempt
    assert delays == [0.01, 0.02]  # exponential backoff: base * 2**attempt


def test_retry_exhausts_and_raises_step_failed_error() -> None:
    def always_fails(attempt: int) -> str:
        raise RetryableStepError(f"attempt {attempt} always fails")

    with pytest.raises(StepFailedError) as exc_info:
        run_step_with_retry(always_fails, max_retries=2, base_delay=0.001, sleep_fn=lambda _: None)

    assert "3 attempt(s)" in str(exc_info.value)  # 1 initial + 2 retries


def test_simulated_fetch_failure_is_retried_by_step_process_source() -> None:
    # fail_times=1: the first attempt raises, the second (attempt=1) succeeds.
    source = {"name": "flaky-source", "content": "Real content with a number: 42.", "fail_times": 1}
    payload = run_step_with_retry(
        lambda attempt: step_process_source(source, StubClient(), attempt),
        max_retries=3,
        base_delay=0.001,
        sleep_fn=lambda _: None,
    )
    assert payload["source"] == "flaky-source"
    assert "42" in payload["summary"] or "42" in " ".join(payload["facts"])


def test_job_fails_cleanly_when_a_source_always_fails(tmp_path) -> None:
    sources = [
        {"name": "good-source", "content": "Reliable content about Acme Corp, founded in 2001."},
        {"name": "bad-source", "content": "Never fetched.", "fail_times": 999},
    ]
    with JobStore(str(tmp_path / "jobs.db")) as store:
        job = store.create_job(sources, max_sources=25)
        finished = execute_job(
            store,
            job.id,
            StubClient(),
            max_retries=2,
            base_delay=0.001,
            max_wall_clock_seconds=30,
            sleep_fn=lambda _: None,
        )

    assert finished.status is JobStatus.FAILED
    assert finished.error  # error is recorded, not silently swallowed
    assert "bad-source" in finished.error

    # Partial progress before the failure is preserved: the good source's
    # checkpoint exists, but nothing for the step that never finished.
    with JobStore(str(tmp_path / "jobs.db")) as store:
        checkpoints = store.get_checkpoints(job.id)
    assert len(checkpoints) == 1
    assert checkpoints[0].step_name == "process_source:good-source"


def test_job_bound_exceeded_fails_the_job(tmp_path) -> None:
    sources = [{"name": "s1", "content": "Some content here."}]
    with JobStore(str(tmp_path / "jobs.db")) as store:
        job = store.create_job(sources, max_sources=25)
        finished = execute_job(
            store,
            job.id,
            StubClient(),
            max_retries=1,
            base_delay=0.001,
            max_wall_clock_seconds=-1,  # already exceeded before the first step -> deterministic
            sleep_fn=lambda _: None,
        )
    assert finished.status is JobStatus.FAILED
    assert "wall clock" in finished.error
