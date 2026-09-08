"""The model-provider seam for this starter.

Two capabilities go through it: `create()` drives one turn of the example
agent's tool loop, and `judge_groundedness()` is the LLM-as-judge metric.
Routing both through the same `AnthropicClient` / `StubClient` seam means
the judge — normally the one metric that costs a real API call — also has
a free, deterministic offline stand-in, so `agent-evals run` is a genuine
CI gate with no secrets required.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from .config import Config, MissingCredentialsError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


Block = TextBlock | ToolUseBlock


@dataclass(frozen=True)
class LLMResponse:
    content: list[Block]
    stop_reason: str


_GROUNDEDNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


class LLMClient(Protocol):
    def create(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str
    ) -> LLMResponse: ...

    def judge_groundedness(
        self, question: str, answer: str, expected_facts: list[str]
    ) -> dict[str, Any]: ...


class AnthropicClient:
    """Wraps `anthropic.Anthropic().messages.create` for both the agent loop
    and the (paid, when used live) groundedness judge."""

    def __init__(self, model: str, max_tokens: int = 4096) -> None:
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def create(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str
    ) -> LLMResponse:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
        except anthropic.AuthenticationError as e:
            raise MissingCredentialsError(
                "Anthropic rejected ANTHROPIC_API_KEY. Check your .env file, or run with --offline."
            ) from e
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"Rate limited by the Anthropic API: {e}") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Anthropic API error ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"Could not reach the Anthropic API: {e}") from e
        return LLMResponse(
            content=_convert_blocks(response.content), stop_reason=response.stop_reason
        )

    def judge_groundedness(
        self, question: str, answer: str, expected_facts: list[str]
    ) -> dict[str, Any]:
        """One structured-output call, `effort: low` — this is the metric that costs money."""
        prompt = _groundedness_prompt(question, answer, expected_facts)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                output_config={
                    "format": {"type": "json_schema", "schema": _GROUNDEDNESS_SCHEMA},
                    "effort": "low",
                },
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AuthenticationError as e:
            raise MissingCredentialsError(
                "Anthropic rejected ANTHROPIC_API_KEY. Check your .env file, or run with --offline."
            ) from e
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"Rate limited by the Anthropic API: {e}") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Anthropic API error ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"Could not reach the Anthropic API: {e}") from e
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        return json.loads(text)


def _groundedness_prompt(question: str, answer: str, expected_facts: list[str]) -> str:
    facts = "; ".join(expected_facts) if expected_facts else "(none specified)"
    return (
        "You are grading whether an AI assistant's answer is grounded in a set of "
        "known facts (not fabricating beyond them, not contradicting them).\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Known facts: {facts}\n\n"
        "Score 0.0-1.0: 1.0 means fully supported by the known facts, 0.0 means "
        "unsupported or contradictory."
    )


def _convert_blocks(blocks: list[Any]) -> list[Block]:
    converted: list[Block] = []
    for block in blocks:
        if block.type == "text":
            converted.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            converted.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
        # Other block types (e.g. thinking) are outside this starter's scope.
    return converted


# --- StubClient: deterministic, offline, keyword-routed ---------------------

_REFUSAL_TRIGGERS = (
    "credit card number",
    "social security",
    "hack into",
    "bypass security",
    "someone's password",
    "password of",
)
_ARITH_OPERATOR_RE = re.compile(r"\d\s*[-+*/]\s*[\d(]")
_ARITH_EXPR_RE = re.compile(r"[0-9(][0-9+\-*/(). ]*[0-9)]")
_REFUSAL_TEXT = "I can't help with that — it falls outside what I'm able to assist with."


class StubClient:
    """Deterministic, network-free stand-in for `AnthropicClient`.

    `create()` picks a tool (or refuses) by keyword-matching the user's
    message, then summarizes whatever tool result it gets on the next
    turn. `judge_groundedness()` is a heuristic keyword-overlap score,
    honestly labeled in its `reasoning` field as a stub — never presented
    as a real evaluation.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def create(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str
    ) -> LLMResponse:
        del tools, system  # the stub routes on keywords, not the schema
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):
            return self._summarize(last["content"])
        return self._choose_action(_first_user_text(messages))

    def _choose_action(self, text: str) -> LLMResponse:
        lowered = text.lower()
        if any(trigger in lowered for trigger in _REFUSAL_TRIGGERS):
            return LLMResponse(content=[TextBlock(text=_REFUSAL_TEXT)], stop_reason="refusal")
        if _ARITH_OPERATOR_RE.search(text):
            expr_match = _ARITH_EXPR_RE.search(text)
            expression = expr_match.group(0).strip() if expr_match else text.strip()
            return self._tool_call("calculator", {"expression": expression})
        return self._tool_call("search_kb", {"query": text.strip(" ?.!")})

    def _summarize(self, tool_results: list[dict[str, Any]]) -> LLMResponse:
        content = tool_results[0].get("content", "")
        text = f"Based on what I found: {content}"
        return LLMResponse(content=[TextBlock(text=text)], stop_reason="end_turn")

    def _tool_call(self, name: str, tool_input: dict[str, Any]) -> LLMResponse:
        self._call_count += 1
        block = ToolUseBlock(id=f"toolu_stub_{self._call_count}", name=name, input=tool_input)
        return LLMResponse(content=[block], stop_reason="tool_use")

    def judge_groundedness(
        self, question: str, answer: str, expected_facts: list[str]
    ) -> dict[str, Any]:
        del question
        if not expected_facts:
            return {
                "score": 1.0,
                "reasoning": (
                    "stub judge (offline heuristic, not a real evaluation): "
                    "no expected facts to check."
                ),
            }
        lowered = answer.lower()
        hits = sum(1 for fact in expected_facts if fact.lower() in lowered)
        score = round(hits / len(expected_facts), 4)
        return {
            "score": score,
            "reasoning": (
                f"stub judge (offline heuristic, not a real evaluation): "
                f"{hits}/{len(expected_facts)} expected facts found verbatim in the answer."
            ),
        }


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], str):
            return message["content"]
    return ""


def get_client(config: Config, offline: bool) -> LLMClient:
    """Pick the LLM client per the shared starter contract.

    `--offline` (`offline=True`) always wins. Otherwise `LLM_PROVIDER`
    decides; if it's unset, default to Anthropic when a key is present and
    fall back to the stub — with one INFO log line — when it isn't.
    """
    if offline:
        logger.info("offline mode requested; using the stub client")
        return StubClient()

    if config.llm_provider == "stub":
        logger.info("LLM_PROVIDER=stub; using the stub client")
        return StubClient()

    if config.llm_provider == "anthropic":
        if not config.anthropic_api_key:
            raise MissingCredentialsError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. Copy "
                ".env.example to .env and add your key from "
                "https://console.anthropic.com/settings/keys, or run with --offline."
            )
        return AnthropicClient(model=config.anthropic_model)

    if config.anthropic_api_key:
        return AnthropicClient(model=config.anthropic_model)

    logger.info(
        "ANTHROPIC_API_KEY not set; falling back to the offline stub client. "
        "Set ANTHROPIC_API_KEY (see .env.example) or pass --offline to silence this message."
    )
    return StubClient()
