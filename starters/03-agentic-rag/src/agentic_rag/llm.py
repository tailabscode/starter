"""The model-provider seam: swap between the real Anthropic client and a deterministic stub.

Every starter in this collection must run end-to-end with no API key. ``get_client()`` decides
which implementation to hand back, defaulting to the stub whenever no key is configured, and
always logging that fallback so it is never silent.
"""

from __future__ import annotations

import logging
from typing import Protocol

import anthropic

from agentic_rag.config import Config
from agentic_rag.errors import MissingCredentialsError
from agentic_rag.schemas import AgentStepResult, QueryPlan, ToolCall
from agentic_rag.stub_logic import stub_agent_step, stub_plan

logger = logging.getLogger("agentic_rag")

_PLAN_SYSTEM = (
    "You are the planning stage of a retrieval agent. Decide whether the user's question needs "
    "corpus retrieval at all, and if so, whether one search suffices or the question requires "
    "finding one fact and then searching again for an entity that fact points at."
)


class LLMClient(Protocol):
    """The two model calls the agentic-rag starter needs."""

    def plan_query(self, question: str, topics: list[dict]) -> QueryPlan: ...

    def agent_step(
        self, system: str, messages: list[dict], tools: list[dict]
    ) -> AgentStepResult: ...


class AnthropicClient:
    """Talks to the real Claude API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def plan_query(self, question: str, topics: list[dict]) -> QueryPlan:
        topic_lines = "\n".join(
            f"- {t['title']} ({t['doc_id']}): {', '.join(t['headings'])}" for t in topics
        )
        user_content = (
            f"Question: {question}\n\nCorpus contents:\n{topic_lines}\n\n"
            "Classify this question and draft the search plan."
        )
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=_PLAN_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            output_format=QueryPlan,
        )
        return response.parsed_output

    def agent_step(self, system: str, messages: list[dict], tools: list[dict]) -> AgentStepResult:
        kwargs = {}
        if tools:
            kwargs["tools"] = tools
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
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
    return AgentStepResult(
        stop_reason=response.stop_reason or "end_turn",
        content=content,
        tool_calls=tool_calls,
        text="\n".join(text_parts),
    )


class StubClient:
    """Deterministic, offline stand-in for :class:`AnthropicClient`. See ``stub_logic.py``."""

    def plan_query(self, question: str, topics: list[dict]) -> QueryPlan:
        return stub_plan(question, topics)

    def agent_step(self, system: str, messages: list[dict], tools: list[dict]) -> AgentStepResult:
        del system  # the stub doesn't need the system prompt; it works off messages + tools
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
