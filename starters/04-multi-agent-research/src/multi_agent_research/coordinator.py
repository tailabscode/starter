"""The coordinator: decomposes a question into a bounded set of independent subtasks."""

from __future__ import annotations

import json

from multi_agent_research.corpus import Corpus
from multi_agent_research.llm import LLMClient
from multi_agent_research.schemas import Subtask, SubtaskPlan

_SYSTEM = (
    "You are the coordinator of a research team. Split the user's question into independent "
    "subtasks that can be researched in parallel by separate researchers, each with access to "
    "a local corpus. Keep subtasks genuinely independent -- no subtask should depend on "
    "another subtask's answer. Never propose more subtasks than the stated maximum."
)


def decompose(question: str, corpus: Corpus, llm: LLMClient, *, max_subtasks: int) -> SubtaskPlan:
    """Decompose ``question`` into at most ``max_subtasks`` independent subtasks.

    Bounded twice: the prompt states the maximum, and the result is truncated again here in
    case the model (or the stub) proposes too many -- the bound is enforced in code, not just
    requested in the prompt.
    """
    user_content = (
        f"Question: {question}\n\n"
        f"Corpus topics (JSON): {json.dumps(corpus.topics())}\n"
        f"Maximum subtasks: {max_subtasks}\n\n"
        "Decompose the question into independent subtasks, each assigned to its own researcher."
    )
    plan = llm.structured(
        system=_SYSTEM, user_content=user_content, schema=SubtaskPlan, thinking=True
    )
    if len(plan.subtasks) > max_subtasks:
        plan = SubtaskPlan(reasoning=plan.reasoning, subtasks=plan.subtasks[:max_subtasks])
    if not plan.subtasks:
        plan = SubtaskPlan(
            reasoning="No subtasks were proposed; falling back to the question as-is.",
            subtasks=[Subtask(id="full-question", description=question)],
        )
    return plan
