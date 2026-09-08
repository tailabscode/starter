"""JobStore: checkpoint persistence round-trips, and status/current_step advance correctly."""

import json

import pytest

from long_running_agent.errors import InvalidTransitionError
from long_running_agent.models import JobStatus
from long_running_agent.store import JobStore

SOURCES = [
    {"name": "s1", "content": "Alpha content."},
    {"name": "s2", "content": "Beta content."},
]


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "jobs.db")
    with JobStore(db_path) as s:
        yield s


def test_create_job_sets_pending_and_total_steps(store: JobStore) -> None:
    job = store.create_job(SOURCES, max_sources=25)
    assert job.status is JobStatus.PENDING
    assert job.current_step == 0
    assert job.total_steps == len(SOURCES) + 1  # +1 for the synthesis step
    assert json.loads(job.input) == SOURCES


def test_checkpoint_round_trips_payload(store: JobStore) -> None:
    job = store.create_job(SOURCES, max_sources=25)
    store.mark_running(job.id)

    payload = {"source": "s1", "summary": "a summary", "facts": ["fact one", "fact two"]}
    store.checkpoint_step(job.id, 0, "process_source:s1", payload)

    checkpoints = store.get_checkpoints(job.id)
    assert len(checkpoints) == 1
    assert checkpoints[0].step_index == 0
    assert checkpoints[0].step_name == "process_source:s1"
    assert json.loads(checkpoints[0].payload_json) == payload  # exact round trip


def test_checkpoint_step_advances_current_step_and_status(store: JobStore) -> None:
    job = store.create_job(SOURCES, max_sources=25)
    store.mark_running(job.id)
    store.checkpoint_step(job.id, 0, "process_source:s1", {"ok": True})

    updated = store.get_job(job.id)
    assert updated.status is JobStatus.CHECKPOINTED
    assert updated.current_step == 1


def test_multiple_checkpoints_persist_in_order(store: JobStore) -> None:
    job = store.create_job(SOURCES, max_sources=25)
    store.mark_running(job.id)
    store.checkpoint_step(job.id, 0, "process_source:s1", {"n": 1})
    store.mark_running(job.id)  # CHECKPOINTED -> RUNNING for the next step
    store.checkpoint_step(job.id, 1, "process_source:s2", {"n": 2})

    checkpoints = store.get_checkpoints(job.id)
    assert [c.step_index for c in checkpoints] == [0, 1]
    assert [json.loads(c.payload_json)["n"] for c in checkpoints] == [1, 2]


def test_checkpoint_step_requires_running_status(store: JobStore) -> None:
    job = store.create_job(SOURCES, max_sources=25)
    # job is still PENDING -- checkpointing before mark_running must fail loudly.
    with pytest.raises(InvalidTransitionError):
        store.checkpoint_step(job.id, 0, "process_source:s1", {"n": 1})


def test_complete_job_sets_result_and_status(store: JobStore) -> None:
    job = store.create_job(SOURCES, max_sources=25)
    store.mark_running(job.id)
    store.checkpoint_step(job.id, 0, "process_source:s1", {"n": 1})
    store.mark_running(job.id)
    store.checkpoint_step(job.id, 1, "process_source:s2", {"n": 2})
    store.mark_running(job.id)
    store.checkpoint_step(job.id, 2, "synthesize", {"synthesis": "done"})

    completed = store.complete_job(job.id, {"synthesis": "done", "source_count": 2})
    assert completed.status is JobStatus.COMPLETED
    assert json.loads(completed.result)["source_count"] == 2


def test_fail_job_records_error_not_silently(store: JobStore) -> None:
    job = store.create_job(SOURCES, max_sources=25)
    store.mark_running(job.id)
    failed = store.fail_job(job.id, "boom: retries exhausted")
    assert failed.status is JobStatus.FAILED
    assert failed.error == "boom: retries exhausted"


def test_too_many_sources_rejected(store: JobStore) -> None:
    from long_running_agent.errors import TooManySourcesError

    too_many = [{"name": f"s{i}", "content": "x"} for i in range(5)]
    with pytest.raises(TooManySourcesError):
        store.create_job(too_many, max_sources=3)
