"""Deterministic scorers. The (optional, LLM-judge) groundedness metric lives
on the `LLMClient` seam itself (see `llm.py`) since it needs to make a call."""

from .example_agent import ToolCall


def score_keyword_coverage(output_text: str, expected_facts: list[str]) -> float:
    """Fraction of `expected_facts` found as a case-insensitive substring of
    `output_text`. An empty `expected_facts` list has nothing to check -> 1.0."""
    if not expected_facts:
        return 1.0
    lowered = output_text.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in lowered)
    return hits / len(expected_facts)


def score_tool_use(tool_calls: list[ToolCall], expected_tool_calls: list[str]) -> float:
    """1.0 if the set of tool names called matches `expected_tool_calls` exactly
    (order-independent) and every call has non-empty argument values, else 0.0.

    An empty `expected_tool_calls` means "expect no tool calls at all".
    """
    called_names = sorted(call.name for call in tool_calls)
    if not expected_tool_calls:
        return 1.0 if not called_names else 0.0
    if called_names != sorted(expected_tool_calls):
        return 0.0
    for call in tool_calls:
        if not call.input:
            return 0.0
        if any(isinstance(v, str) and not v.strip() for v in call.input.values()):
            return 0.0
    return 1.0
