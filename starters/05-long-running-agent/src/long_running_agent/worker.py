"""Background execution: a simple polling runner that claims and advances jobs.

Two modes, both bounded so nothing loops forever by accident:

- Drain mode (default): claim and run every currently PENDING/CHECKPOINTED
  job, then exit. This is what makes `--offline` demos work in one terminal
  -- no second process needs to be kept running.
- Continuous mode (`--continuous`): keep polling for new jobs, bounded by
  `max_iterations` (required, so a forgotten flag can't loop forever).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .llm import LLMClient
from .runner import execute_job
from .store import JobStore

logger = logging.getLogger(__name__)


def worker_loop(
    store: JobStore,
    llm_client: LLMClient,
    *,
    max_retries: int,
    base_delay: float,
    max_wall_clock_seconds: float,
    continuous: bool = False,
    poll_interval: float = 1.0,
    max_iterations: int | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Poll for claimable jobs and run each to completion/failure. Returns jobs processed."""
    if continuous and max_iterations is None:
        max_iterations = 60  # hard default ceiling for continuous mode

    processed = 0
    idle_iterations = 0
    while max_iterations is None or idle_iterations + processed < max_iterations:
        job = store.claim_next_pending()
        if job is None:
            if not continuous:
                break
            idle_iterations += 1
            sleep_fn(poll_interval)
            continue

        logger.info("worker claimed job", extra={"job_id": job.id, "status": job.status.value})
        finished = execute_job(
            store,
            job.id,
            llm_client,
            max_retries=max_retries,
            base_delay=base_delay,
            max_wall_clock_seconds=max_wall_clock_seconds,
            sleep_fn=sleep_fn,
        )
        logger.info(
            "worker finished job", extra={"job_id": job.id, "status": finished.status.value}
        )
        processed += 1

    return processed
