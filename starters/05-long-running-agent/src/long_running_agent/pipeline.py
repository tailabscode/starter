"""The concrete long-running job: multi-source research + roll-up synthesis.

For each source: fetch (simulated locally, no network), summarize with the
model, extract key facts. Once every source is processed, a final step
synthesizes a roll-up across all of them. Each of these is one resumable,
checkpointed step -- see `runner.execute_job`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from .errors import RetryableStepError, SourceFetchError, StepFailedError
from .llm import LLMCallError, LLMClient

T = TypeVar("T")

# Small, synthetic, obviously-fictional sample sources. `submit` uses this as
# the default input when no --sources-file/--sources-json is given, so the
# starter is runnable with zero setup. See data/sources.json for the same
# content in a human-inspectable file (used by the README's example).
DEFAULT_SOURCES: list[dict] = [
    {
        "name": "source-alpha",
        "url": "https://example.invalid/meridian/overview",
        "content": (
            "Project Meridian is a fictional research initiative started in 2019 by "
            "Northwind Labs. It combines 3 subsystems: a scheduler, a ledger, and a "
            "notification bus. By 2022 the project reported 12 pilot deployments "
            "across 4 simulated regions."
        ),
    },
    {
        "name": "source-beta",
        "url": "https://example.invalid/meridian/architecture",
        "content": (
            "The Meridian scheduler assigns work in rounds of 5 minutes. Northwind "
            "Labs measured a median round latency of 340 milliseconds in their "
            "synthetic benchmark. The ledger subsystem retains 90 days of history "
            "by default."
        ),
    },
    {
        "name": "source-gamma",
        "url": "https://example.invalid/meridian/incidents",
        "content": (
            "Between 2021 and 2023, Meridian's test environment recorded 7 simulated "
            "incidents, none rated above severity 3. The notification bus was the "
            "root cause in 4 of those 7 cases, prompting a rewrite in early 2023."
        ),
    },
    {
        "name": "source-delta",
        "url": "https://example.invalid/meridian/roadmap",
        "content": (
            "Northwind Labs' fictional roadmap for Meridian lists 2 goals for the "
            "next cycle: reduce median round latency below 200 milliseconds, and "
            "extend ledger retention to 180 days. Both are marked as stretch goals."
        ),
    },
]


def run_step_with_retry(
    step_fn: Callable[[int], T],
    *,
    max_retries: int,
    base_delay: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Call `step_fn(attempt)` (attempt starting at 0), retrying on RetryableStepError.

    Exponential backoff: delay = base_delay * 2**attempt, up to `max_retries`
    retries. On final exhaustion, raises StepFailedError wrapping the last
    error -- the caller (runner.execute_job) turns that into a FAILED job
    with the error recorded, never a silent swallow.
    """
    attempt = 0
    while True:
        try:
            return step_fn(attempt)
        except RetryableStepError as exc:
            if attempt >= max_retries:
                raise StepFailedError(f"step failed after {attempt + 1} attempt(s): {exc}") from exc
            sleep_fn(base_delay * (2**attempt))
            attempt += 1


def simulate_fetch(source: dict, attempt: int) -> str:
    """Simulated network fetch -- no real HTTP call, deterministic per attempt.

    `source["fail_times"]` (default 0) lets a source rehearse transient
    failures: the fetch raises for the first `fail_times` attempts, then
    succeeds. This is how the retry/backoff and crash-resume tests drive
    failures deterministically, without mocking the network.
    """
    fail_times = int(source.get("fail_times", 0))
    if attempt < fail_times:
        raise SourceFetchError(
            f"simulated transient fetch failure for {source['name']!r} "
            f"(attempt {attempt + 1}/{fail_times} configured failures)"
        )
    content = source.get("content")
    if not content:
        raise SourceFetchError(f"source {source['name']!r} has no content to fetch")
    return content


def step_process_source(source: dict, llm_client: LLMClient, attempt: int) -> dict:
    """fetch -> summarize -> extract facts, for one source. One resumable step."""
    content = simulate_fetch(source, attempt)
    try:
        summary = llm_client.summarize(source["name"], content)
        facts = llm_client.extract_facts(source["name"], content)
    except LLMCallError as exc:
        raise RetryableStepError(f"model call failed for {source['name']!r}: {exc}") from exc
    return {
        "source": source["name"],
        "url": source.get("url", ""),
        "summary": summary,
        "facts": facts,
    }


def step_synthesize(per_source_results: list[dict], llm_client: LLMClient, attempt: int) -> dict:
    """Final roll-up synthesis across every processed source. The last step."""
    try:
        synthesis = llm_client.synthesize(per_source_results)
    except LLMCallError as exc:
        raise RetryableStepError(f"synthesis model call failed: {exc}") from exc
    return {"synthesis": synthesis, "source_count": len(per_source_results)}
