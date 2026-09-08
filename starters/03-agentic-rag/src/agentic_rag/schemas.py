"""Structured-output and trace data shapes shared across the planner, agent loop, and CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

Classification = Literal["NO_RETRIEVAL", "SINGLE_LOOKUP", "MULTI_HOP"]


class QueryPlan(BaseModel):
    """The output of the planning call: how (if at all) to approach retrieval."""

    classification: Classification = Field(
        description=(
            "NO_RETRIEVAL: answerable from general knowledge, nothing in the corpus is needed. "
            "SINGLE_LOOKUP: one search should find the answer. "
            "MULTI_HOP: the answer requires finding one fact, then searching again for an "
            "entity that fact points at."
        )
    )
    reasoning: str = Field(description="One or two sentences explaining the classification.")
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Search queries to run, in order. Empty for NO_RETRIEVAL.",
    )


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class AgentStepResult:
    """One turn of the agent loop, normalized across the real and stub clients.

    ``content`` is a list of plain dicts in Anthropic content-block shape (``text`` /
    ``tool_use``) so it can be echoed straight back into the next request's message history
    regardless of which client produced it.
    """

    stop_reason: str
    content: list[dict]
    tool_calls: list[ToolCall]
    text: str


@dataclass
class TraceStep:
    """One entry in the machine-readable decision trace."""

    step: int
    kind: str  # "plan" | "tool_call" | "final" | "bound_hit"
    detail: dict


@dataclass
class RunTrace:
    """The full record of one query: the plan, every tool call, and why the run stopped."""

    query: str
    plan: dict = field(default_factory=dict)
    steps: list[TraceStep] = field(default_factory=list)
    stopped_reason: str = ""
    citations: list[str] = field(default_factory=list)
    answer: str = ""

    def add_step(self, kind: str, detail: dict) -> None:
        self.steps.append(TraceStep(step=len(self.steps) + 1, kind=kind, detail=detail))

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload
