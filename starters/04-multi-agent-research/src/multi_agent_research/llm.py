"""The model-provider seam: swap between the real Anthropic client and a deterministic stub.

Every starter in this collection must run end-to-end with no API key. ``get_client()`` decides
which implementation to hand back, defaulting to the stub whenever no key is configured, and
always logging that fallback so it is never silent.
"""

from __future__ import annotations

import logging
from typing import Protocol

import anthropic
from pydantic import BaseModel

from multi_agent_research.config import Config
from multi_agent_research.errors import MissingCredentialsError
from multi_agent_research.schemas import (
    AgentStepResult,
    Critique,
    SubtaskPlan,
    SynthesisReport,
    ToolCall,
)
from multi_agent_research.stub_logic import (
    stub_agent_step,
    stub_critique,
    stub_decompose,
    stub_synthesize,
)

logger = logging.getLogger("multi_agent_research")


class LLMClient(Protocol):
    """The two model calls this starter needs: structured output, and a tool-calling turn."""

    def structured(
        self, *, system: str, user_content: str, schema: type[BaseModel], thinking: bool = True
    ) -> BaseModel: ...

    def agent_step(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        effort: str = "high",
        max_tokens: int = 4096,
    ) -> AgentStepResult: ...


class AnthropicClient:
    """Talks to the real Claude API. Thread-safe: researchers share one instance."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def structured(
        self, *, system: str, user_content: str, schema: type[BaseModel], thinking: bool = True
    ) -> BaseModel:
        kwargs = {"thinking": {"type": "adaptive"}} if thinking else {}
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                output_format=schema,
                **kwargs,
            )
        except anthropic.AuthenticationError as exc:
            raise MissingCredentialsError(
                "ANTHROPIC_API_KEY was rejected by the API. Check the key at "
                "https://console.anthropic.com/settings/keys, or run with --offline."
            ) from exc
        return response.parsed_output

    def agent_step(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        effort: str = "high",
        max_tokens: int = 4096,
    ) -> AgentStepResult:
        kwargs: dict = {"output_config": {"effort": effort}}
        if tools:
            kwargs["tools"] = tools
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                **kwargs,
            )
        except anthropic.AuthenticationError as exc:
            raise MissingCredentialsError(
                "ANTHROPIC_API_KEY was rejected by the API. Check the key at "
                "https://console.anthropic.com/settings/keys, or run with --offline."
            ) from exc
        return _normalize_response(response)


def _normalize_response(response) -> AgentStepResult:  # noqa: ANN001 - anthropic SDK type
    content: list[dict] = []
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
            text_parts.append(block.text)
        elif block.type == "tool_use":
            content.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
            )
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
        elif block.type == "thinking":
            continue  # not replayed; see agentic-rag's README for the same design choice
        else:
            # Passthrough for server-tool blocks (e.g. web_search_tool_result) we don't act on
            # ourselves but still need to round-trip so the model sees its own prior turn.
            content.append(block.model_dump())
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
    }
    return AgentStepResult(
        stop_reason=response.stop_reason or "end_turn",
        content=content,
        tool_calls=tool_calls,
        text="\n".join(text_parts),
        usage=usage,
    )


class StubClient:
    """Deterministic, offline stand-in for :class:`AnthropicClient`. See ``stub_logic.py``."""

    def structured(
        self, *, system: str, user_content: str, schema: type[BaseModel], thinking: bool = True
    ) -> BaseModel:
        del system, thinking
        if schema is SubtaskPlan:
            return stub_decompose(user_content)
        if schema is Critique:
            return stub_critique(user_content)
        if schema is SynthesisReport:
            return stub_synthesize(user_content)
        raise ValueError(f"StubClient has no heuristic for schema {schema!r}")

    def agent_step(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        effort: str = "high",
        max_tokens: int = 4096,
    ) -> AgentStepResult:
        del system, effort, max_tokens
        return stub_agent_step(messages, tools)


def get_client(config: Config, *, offline: bool = False) -> LLMClient:
    """Pick a client by config, defaulting to the stub whenever no key is available."""
    if offline:
        logger.info("Using StubClient: --offline flag was passed.")
        return StubClient()

    provider = config.llm_provider or ("anthropic" if config.anthropic_api_key else "stub")

    if provider == "stub":
        if not config.llm_provider:
            logger.info(
                "Using StubClient: ANTHROPIC_API_KEY is not set. Set it (see .env.example) or "
                "set LLM_PROVIDER=anthropic to use the real model."
            )
        return StubClient()

    if provider == "anthropic":
        if not config.anthropic_api_key:
            raise MissingCredentialsError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key from "
                "https://console.anthropic.com/settings/keys, or run with --offline."
            )
        return AnthropicClient(api_key=config.anthropic_api_key, model=config.model)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'anthropic' or 'stub')")
