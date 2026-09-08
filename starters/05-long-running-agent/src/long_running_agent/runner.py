"""Drives one job from its current checkpoint to COMPLETED or FAILED.

This is the single execution path used by both the synchronous foreground
mode (`submit --run`, `resume`) and the background `worker` polling loop --
there is exactly one place that knows how to advance a job.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable

from .errors import JobBoundExceededError, JobNotFoundError, StepFailedError
from .llm import LLMClient
from .models import Job, JobStatus
from .pipeline import run_step_with_retry, step_process_source, step_synthesize
from .store import JobStore

logger = logging.getLogger(__name__)


def execute_job(
    store: JobStore,
    job_id: str,
    llm_client: LLMClient,
    *,
    max_retries: int,
    base_delay: float,
    max_wall_clock_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    simulate_crash_after_step: int | None = None,
    clock_fn: Callable[[], float] = time.monotonic,
) -> Job:
    """Run `job_id` to completion (or failure), skipping already-checkpointed steps.

    Resumability: on entry we read the job's `current_step` (the index of the
    next step to run) and its existing checkpoints, and simply never touch
    steps below `current_step` again -- there is no "did we already do this"
    branch to get wrong, because the loop's start point *is* the proof.

    `simulate_crash_after_step`, if set, calls `os._exit(1)` immediately after
    that step's checkpoint is durably committed -- simulating a hard process
    kill (not a graceful shutdown) so the demo genuinely proves durability
    comes from SQLite, not from in-memory state or cleanup code.
    """
    job = store.get_job(job_id)
    if job is None:
        raise JobNotFoundError(f"no job with id {job_id!r}")
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        logger.info(
            "job already finished, nothing to do",
            extra={"job_id": job_id, "status": job.status.value},
        )
        return job

    sources = json.loads(job.input)
    start = clock_fn()

    # Rebuild the per-source results already on disk, so a resumed synthesis
    # step sees every source, not just the ones run in *this* process. Also
    # recover a synthesis result already on disk, in case a previous process
    # crashed between committing the last checkpoint and finalizing the job.
    per_source_results: list[dict] = []
    final_result: dict | None = None
    for checkpoint in store.get_checkpoints(job_id):
        if checkpoint.step_name.startswith("process_source:"):
            per_source_results.append(json.loads(checkpoint.payload_json))
        elif checkpoint.step_name == "synthesize":
            final_result = json.loads(checkpoint.payload_json)

    step_index = job.current_step
    total_steps = job.total_steps

    try:
        while step_index < total_steps:
            # PENDING/CHECKPOINTED -> RUNNING for this step; a no-op if a crash
            # left the job RUNNING already (nothing to skip past in that case).
            # Done before the bound check so a bound failure always has a
            # RUNNING job to fail, never a PENDING/CHECKPOINTED one.
            store.mark_running(job_id)

            if clock_fn() - start > max_wall_clock_seconds:
                raise JobBoundExceededError(
                    f"job exceeded max wall clock of {max_wall_clock_seconds}s "
                    f"at step {step_index}/{total_steps}"
                )

            is_synthesis_step = step_index == len(sources)
            if is_synthesis_step:
                step_name = "synthesize"
                payload = run_step_with_retry(
                    lambda attempt: step_synthesize(per_source_results, llm_client, attempt),
                    max_retries=max_retries,
                    base_delay=base_delay,
                    sleep_fn=sleep_fn,
                )
                final_result = payload
            else:
                source = sources[step_index]
                step_name = f"process_source:{source['name']}"
                payload = run_step_with_retry(
                    lambda attempt, source=source: step_process_source(source, llm_client, attempt),
                    max_retries=max_retries,
                    base_delay=base_delay,
                    sleep_fn=sleep_fn,
                )
                per_source_results.append(payload)

            logger.info(
                "step checkpointed",
                extra={"job_id": job_id, "step_index": step_index, "step_name": step_name},
            )
            store.checkpoint_step(job_id, step_index, step_name, payload)
            step_index += 1

            just_finished_step = step_index - 1
            if just_finished_step == simulate_crash_after_step:
                # os._exit skips buffer flushing (unlike sys.exit/a normal
                # crash reaching the OS after a SIGKILL usually does for a
                # line-buffered terminal) -- flush explicitly so callers
                # capturing stdout/stderr (e.g. a subprocess pipe) still see
                # everything printed before the kill.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)  # intentional hard process kill, for the crash/resume demo and test

        store.complete_job(job_id, final_result)
    except StepFailedError as exc:
        store.fail_job(job_id, str(exc))
        logger.error("job failed: retries exhausted", extra={"job_id": job_id, "error": str(exc)})
    except JobBoundExceededError as exc:
        store.fail_job(job_id, str(exc))
        logger.error("job failed: bound exceeded", extra={"job_id": job_id, "error": str(exc)})
    except Exception as exc:  # job-boundary fault isolation: recorded on the job, never swallowed
        store.fail_job(job_id, f"unexpected error: {exc}")
        logger.error("job failed: unexpected error", extra={"job_id": job_id, "error": str(exc)})

    return store.get_job(job_id)  # type: ignore[return-value]
