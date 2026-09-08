"""Concurrent researcher dispatch: every subtask gets a finding, and it happens in parallel."""

from __future__ import annotations

import time

from multi_agent_research.llm import StubClient
from multi_agent_research.researcher import run_researchers_concurrently
from multi_agent_research.schemas import AgentStepResult, Subtask


def test_concurrent_dispatch_returns_a_finding_for_every_subtask(corpus) -> None:
    subtasks = [
        Subtask(id=t["doc_id"], description=f"Research {t['title']}") for t in corpus.topics()
    ]

    findings = run_researchers_concurrently(subtasks, llm=StubClient(), corpus=corpus, max_steps=3)

    assert {f["subtask_id"] for f in findings} == {s.id for s in subtasks}
    assert len(findings) == len(subtasks)


def test_each_finding_has_an_answer_and_sources(corpus) -> None:
    subtasks = [Subtask(id="mining", description="Research cobalt mining risk")]

    findings = run_researchers_concurrently(subtasks, llm=StubClient(), corpus=corpus, max_steps=3)

    assert len(findings) == 1
    assert findings[0]["answer"]
    assert findings[0]["sources"]


def test_researchers_actually_run_concurrently_not_sequentially(corpus) -> None:
    """Five subtasks x two slow LLM calls each: concurrent should be far faster than serial."""
    subtasks = [
        Subtask(id=t["doc_id"], description=f"Research {t['title']}") for t in corpus.topics()
    ]

    class SlowStubClient(StubClient):
        def agent_step(self, **kwargs) -> AgentStepResult:  # noqa: ANN003
            time.sleep(0.05)
            return super().agent_step(**kwargs)

    start = time.monotonic()
    findings = run_researchers_concurrently(
        subtasks, llm=SlowStubClient(), corpus=corpus, max_steps=3
    )
    elapsed = time.monotonic() - start

    assert len(findings) == len(subtasks)
    # Serial would take at least 2 calls/subtask * 0.05s * 5 subtasks = 0.5s; concurrent execution
    # across threads should land well under that.
    assert elapsed < 0.35
