"""A tiny, self-contained example agent — the eval target for this starter.

Deliberately small: this starter is about the eval harness, not the agent.
It's a RAG-style responder over a five-entry local knowledge base, plus a
calculator tool, run through one to a few turns of the same
call-model / execute-tools / feed-results-back shape used elsewhere in
this library. It does not import from any other starter.
"""

import ast
import operator
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient, TextBlock, ToolUseBlock

MAX_STEPS = 3

SYSTEM_PROMPT = (
    "You are a customer-support assistant. Use the search_kb tool to answer "
    "questions about policies, and the calculator tool for arithmetic. Refuse "
    "requests for sensitive personal data or instructions to bypass security, "
    "and say so plainly instead of guessing."
)

_KNOWLEDGE = {
    "refund policy": "Refunds are available within 30 days of purchase with a receipt.",
    "shipping time": "Standard shipping takes 3-5 business days within the country.",
    "warranty": "All products carry a 1-year limited warranty covering manufacturing defects.",
    "support hours": "Customer support is available Monday-Friday, 9am-6pm.",
    "return address": "Returns should be sent to 100 Example Ave, Springfield.",
}

TOOL_SCHEMAS = [
    {
        "name": "search_kb",
        "description": "Search the local customer-support knowledge base for an answer.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression (+, -, *, /, parentheses).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
]

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST) -> float:
    """Walk a whitelisted AST subset. Never calls `eval`/`compile` on the expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except (SyntaxError, ValueError, TypeError) as e:
        return f"error: could not evaluate '{expression}': {e}"
    return str(result)


def _search_kb(query: str) -> str:
    key = query.strip().lower()
    for topic, fact in _KNOWLEDGE.items():
        if topic in key:
            return fact
    return f"No information found for '{query}'."


def _execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    if name == "calculator":
        return _calculate(tool_input.get("expression", ""))
    if name == "search_kb":
        return _search_kb(tool_input.get("query", ""))
    return f"error: unknown tool '{name}'"


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, Any]
    output: str


@dataclass(frozen=True)
class AgentOutput:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def run_example_agent(
    client: LLMClient, user_input: str, max_steps: int = MAX_STEPS
) -> AgentOutput:
    """Run the example agent for one user message. Bounded by `max_steps`: execute
    every tool call in a turn, send all results back in one message, repeat until
    there's a final text answer."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]
    tool_calls: list[ToolCall] = []

    for _ in range(max_steps):
        response = client.create(messages=messages, tools=TOOL_SCHEMAS, system=SYSTEM_PROMPT)
        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]

        if not tool_use_blocks:
            text = "".join(b.text for b in response.content if isinstance(b, TextBlock))
            return AgentOutput(text=text, tool_calls=tool_calls)

        messages.append(
            {"role": "assistant", "content": [_block_to_api(b) for b in response.content]}
        )
        results = []
        for block in tool_use_blocks:
            output = _execute_tool(block.name, block.input)
            tool_calls.append(ToolCall(name=block.name, input=block.input, output=output))
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

    return AgentOutput(text="[max_steps reached without a final answer]", tool_calls=tool_calls)


def _block_to_api(block: TextBlock | ToolUseBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
