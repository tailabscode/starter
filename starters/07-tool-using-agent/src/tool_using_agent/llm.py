"""The model-provider seam: `AnthropicClient` for real calls, `StubClient` for
fully offline, deterministic demos and tests. `get_client()` picks between
them so the rest of the codebase never imports `anthropic` directly.
"""

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


class LLMClient(Protocol):
    def create(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str
    ) -> LLMResponse: ...


class AnthropicClient:
    """Wraps `anthropic.Anthropic().messages.create` for the agent loop."""

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
                "Anthropic rejected ANTHROPIC_API_KEY. Check the value in your .env file, "
                "or run with --offline."
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


def _convert_blocks(blocks: list[Any]) -> list[Block]:
    converted: list[Block] = []
    for block in blocks:
        if block.type == "text":
            converted.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            converted.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
        # Other block types (e.g. thinking) are outside this starter's scope.
    return converted


# --- StubClient: deterministic, offline, keyword-routed --------------------

_WEATHER_RE = re.compile(r"weather.*?\bin\s+([A-Za-z\s]+?)(?:[?.!]|$)", re.IGNORECASE)
_CONVERT_RE = re.compile(
    r"convert\s+(-?[\d.]+)\s*([a-zA-Z°]+)\s+(?:to|into)\s+([a-zA-Z°]+)", re.IGNORECASE
)
_ARITH_OPERATOR_RE = re.compile(r"\d\s*[-+*/]\s*[\d(]")
_ARITH_EXPR_RE = re.compile(r"[0-9(][0-9+\-*/(). ]*[0-9)]")
_FACT_TRIGGERS = ("tell me about ", "what is ", "what's ", "who is ", "fact about ")


class StubClient:
    """Deterministic, network-free stand-in for `AnthropicClient`.

    Picks a tool by keyword-matching the user's message (see the module
    docstring in `cli.py` for the exact rules), then on the next turn
    summarizes whatever tool result it was given. This is a demo/test
    double — plausible, useful output derived from real input — never a
    real model, and it is documented as such in the README.
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
        return self._choose_tool(_first_user_text(messages))

    def _choose_tool(self, text: str) -> LLMResponse:
        weather_match = _WEATHER_RE.search(text)
        if "weather" in text.lower() and weather_match:
            city = weather_match.group(1).strip()
            return self._tool_call("http_style_adapter", {"city": city})

        convert_match = _CONVERT_RE.search(text)
        if convert_match:
            value, from_unit, to_unit = convert_match.groups()
            return self._tool_call(
                "unit_convert",
                {"value": float(value), "from_unit": from_unit, "to_unit": to_unit},
            )

        if _ARITH_OPERATOR_RE.search(text):
            expr_match = _ARITH_EXPR_RE.search(text)
            expression = expr_match.group(0).strip() if expr_match else text.strip()
            return self._tool_call("calculator", {"expression": expression})

        return self._tool_call("lookup_fact", {"topic": _extract_topic(text)})

    def _summarize(self, tool_results: list[dict[str, Any]]) -> LLMResponse:
        first = tool_results[0]
        content = first.get("content", "")
        if first.get("is_error"):
            text = f"I couldn't complete that: {content}"
        else:
            text = f"Here's what I found: {content}"
        return LLMResponse(content=[TextBlock(text=text)], stop_reason="end_turn")

    def _tool_call(self, name: str, tool_input: dict[str, Any]) -> LLMResponse:
        self._call_count += 1
        block = ToolUseBlock(id=f"toolu_stub_{self._call_count}", name=name, input=tool_input)
        return LLMResponse(content=[block], stop_reason="tool_use")


def _extract_topic(text: str) -> str:
    lowered = text.strip().lower()
    for trigger in _FACT_TRIGGERS:
        if trigger in lowered:
            return lowered.split(trigger, 1)[1].strip(" ?.!")
    return text.strip(" ?.!")


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], str):
            return message["content"]
    return ""


def get_client(config: Config, offline: bool) -> LLMClient:
    """Pick the LLM client per the shared starter contract.

    `--offline` (i.e. `offline=True`) always wins. Otherwise `LLM_PROVIDER`
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
