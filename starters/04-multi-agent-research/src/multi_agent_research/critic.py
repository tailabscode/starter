"""The critic: reviews collected findings against the original question."""

from __future__ import annotations

import json

from multi_agent_research.corpus import Corpus
from multi_agent_research.llm import LLMClient
from multi_agent_research.schemas import Critique

_SYSTEM = (
    "You are the critic on a research team. Review the findings collected so far against the "
    "original question. Decide whether they are sufficient to write a complete, well-sourced "
    "answer. If not, name the specific missing pieces of information as concrete gaps -- not "
    "vague dissatisfaction -- so researchers can be re-dispatched to close exactly those gaps."
)


def review_findings(
    question: str, findings: list[dict], corpus: Corpus, llm: LLMClient
) -> Critique:
    """Ask the critic whether ``findings`` are sufficient to answer ``question``."""
    user_content = (
        f"Question: {question}\n\n"
        f"All corpus topics (JSON): {json.dumps(corpus.topics())}\n"
        f"Findings so far (JSON): {json.dumps(findings)}\n\n"
        "Decide whether these findings are sufficient. If not, list specific gaps."
    )
    return llm.structured(system=_SYSTEM, user_content=user_content, schema=Critique, thinking=True)
