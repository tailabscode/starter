"""The orchestrator: runs one full research pipeline and persists state after every stage.

coordinator -> researchers (concurrent) -> critic -> [revise for named gaps]* -> synthesiser,
bounded by ``config.max_rounds``. State is written to disk after every stage, so ``show
<run-id>`` can display exactly how far a run got even if a later stage raises.
"""

from __future__ import annotations

import logging
import uuid

from multi_agent_research.config import Config
from multi_agent_research.coordinator import decompose
from multi_agent_research.corpus import Corpus
from multi_agent_research.critic import review_findings
from multi_agent_research.llm import LLMClient
from multi_agent_research.researcher import run_researchers_concurrently
from multi_agent_research.schemas import Subtask
from multi_agent_research.state import ResearchState
from multi_agent_research.synthesizer import synthesize

logger = logging.getLogger("multi_agent_research")


def _merge_findings(existing: list[dict], new: list[dict]) -> list[dict]:
    """Fold ``new`` findings into ``existing``, keyed by subtask id, preserving first-seen order."""
    by_id = {finding["subtask_id"]: finding for finding in existing}
    by_id.update({finding["subtask_id"]: finding for finding in new})
    return list(by_id.values())


def run_research(question: str, *, llm: LLMClient, corpus: Corpus, config: Config) -> ResearchState:
    """Run the full pipeline for ``question`` and return the final, persisted state.

    Bounded by ``config.max_rounds``: if the critic is still unsatisfied when the cap is hit,
    the run terminates cleanly (``termination_reason="max_rounds_reached"``) and synthesises a
    report from whatever findings exist, rather than looping forever chasing a perfect critique.
    """
    run_id = uuid.uuid4().hex[:12]
    state = ResearchState(run_id=run_id, question=question)
    state.save(config.runs_dir)

    plan = decompose(question, corpus, llm, max_subtasks=config.max_subtasks)
    state.plan = plan.model_dump()
    state.save(config.runs_dir)

    subtasks: list[Subtask] = plan.subtasks
    findings: list[dict] = []

    for round_num in range(1, config.max_rounds + 1):
        state.revision_round = round_num
        state.subtasks = [s.model_dump() for s in subtasks]
        state.save(config.runs_dir)

        new_findings = run_researchers_concurrently(
            subtasks,
            llm=llm,
            corpus=corpus,
            max_steps=config.researcher_max_steps,
            enable_web_search=config.enable_web_search,
        )
        for finding in new_findings:
            state.add_usage(finding.get("usage", {}))
        findings = _merge_findings(findings, new_findings)
        state.findings = findings
        state.save(config.runs_dir)

        critique = review_findings(question, findings, corpus, llm)
        state.critiques.append(critique.model_dump())
        state.save(config.runs_dir)

        if critique.is_sufficient:
            state.termination_reason = "sufficient"
            break

        if round_num == config.max_rounds:
            state.termination_reason = "max_rounds_reached"
            logger.warning(
                "Hit max_rounds with unresolved gaps; synthesizing with what was found.",
                extra={"run_id": run_id, "max_rounds": config.max_rounds, "gaps": critique.gaps},
            )
            break

        if not critique.gaps:
            # Insufficient but nothing concrete to act on -- another round would be identical.
            state.termination_reason = "no_gaps_named"
            break

        subtasks = [
            Subtask(id=f"gap-{round_num + 1}-{i}", description=gap)
            for i, gap in enumerate(critique.gaps)
        ]

    report = synthesize(question, findings, llm)
    state.final_report = report.report
    state.save(config.runs_dir)
    return state
