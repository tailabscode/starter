"""Structured-output and message-block data shapes shared across the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class Subtask(BaseModel):
    """One independent piece of the question, dispatched to its own researcher."""

    id: str = Field(description="A short slug, e.g. 'mining-risk'.")
    description: str = Field(description="What this researcher should find out.")


class SubtaskPlan(BaseModel):
    """The coordinator's decomposition of the question."""

    reasoning: str = Field(description="One or two sentences on how the question was split.")
    subtasks: list[Subtask]


class Critique(BaseModel):
    """The critic's structured review of the collected findings."""

    is_sufficient: bool = Field(
        description="True if the findings fully answer the original question."
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Specific missing information, one item per gap. Empty if sufficient.",
    )
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


class SynthesisReport(BaseModel):
    """The synthesiser's final answer, with citations traced back to researcher findings."""

    report: str = Field(
        description="The final answer to the original question, citing sources as [doc_id]."
    )
    citations: list[str] = Field(
        default_factory=list,
        description="doc_ids cited in the report; each must appear in a finding's sources.",
    )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class AgentStepResult:
    """One turn of a researcher's bounded tool loop, normalized across real and stub clients."""

    stop_reason: str
    content: list[dict]
    tool_calls: list[ToolCall]
    text: str
    usage: dict[str, int]
