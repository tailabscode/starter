"""Shared tool contract.

Every tool module in this package exports `SCHEMA` (an Anthropic tool
definition dict, `strict: True`) and a `run(tool_input: dict) -> str`
function. `run` validates its input itself — the JSON schema only checks
shape, not business rules like "unit exists in the table" or "no division
by zero" — and raises `ToolError` on any failure. The agent loop (see
`agent.py`) catches `ToolError` and turns it into a `tool_result` with
`is_error: True`, so a bad tool call is reported to Claude instead of
crashing the loop.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """A tool's own input validation or execution failed. Caught by the agent loop."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: dict[str, Any]
    run: Callable[[dict[str, Any]], str]
