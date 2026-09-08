"""The bounded manual agentic loop.

This intentionally hand-rolls the `while` loop instead of using the SDK's
beta tool runner: the point of this starter is to show the loop mechanics
explicitly — including bounding iterations with `max_steps` and returning
every parallel `tool_result` from one turn in a single user message.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient, TextBlock, ToolUseBlock
from .tools import ToolError, ToolSpec

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You are a careful assistant with access to a small set of tools: a "
    "calculator, a unit converter, a local fact lookup, and a synthetic "
    "weather adapter. Use a tool whenever it would give a more reliable "
    "answer than reasoning alone. If a tool call fails, explain the error "
    "to the user instead of guessing."
)


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool


@dataclass(frozen=True)
class AgentResult:
    final_text: str
    steps_used: int
    hit_max_steps: bool
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


def run_agent(
    client: LLMClient,
    tools: dict[str, ToolSpec],
    user_message: str,
    system: str = SYSTEM_PROMPT,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentResult:
    """Run the tool-use loop for one user message, bounded by `max_steps`.

    Each step: call the model, execute every `tool_use` block it returned
    (there may be several — a parallel tool call), and send all the
    resulting `tool_result` blocks back in a single user message. Stops as
    soon as a turn has no tool calls, or after `max_steps` turns.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    tool_schemas = [spec.schema for spec in tools.values()]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    tool_calls: list[ToolCallRecord] = []

    for step in range(1, max_steps + 1):
        response = client.create(messages=messages, tools=tool_schemas, system=system)
        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]

        if not tool_use_blocks:
            final_text = "".join(b.text for b in response.content if isinstance(b, TextBlock))
            return AgentResult(
                final_text=final_text, steps_used=step, hit_max_steps=False, tool_calls=tool_calls
            )

        messages.append(
            {"role": "assistant", "content": [_block_to_api(b) for b in response.content]}
        )

        tool_results = []
        for block in tool_use_blocks:
            spec = tools.get(block.name)
            if spec is None:
                output = f"Unknown tool '{block.name}'. Available tools: {', '.join(tools)}."
                is_error = True
                logger.warning("unknown tool requested", extra={"tool_name": block.name})
            else:
                try:
                    output = spec.run(block.input)
                    is_error = False
                except ToolError as e:
                    output = str(e)
                    is_error = True
                    logger.info(
                        "tool call failed", extra={"tool_name": block.name, "error": output}
                    )
            tool_results.append(_result_block(block.id, output, is_error))
            tool_calls.append(ToolCallRecord(block.name, block.input, output, is_error))

        messages.append({"role": "user", "content": tool_results})

    logger.warning("max_steps reached", extra={"max_steps": max_steps})
    return AgentResult(
        final_text=(
            f"Reached max_steps ({max_steps}) without a final answer. "
            "Try a simpler request or raise --max-steps."
        ),
        steps_used=max_steps,
        hit_max_steps=True,
        tool_calls=tool_calls,
    )


def _block_to_api(block: TextBlock | ToolUseBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}


def _result_block(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block
