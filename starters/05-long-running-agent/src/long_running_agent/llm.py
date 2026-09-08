"""Model-provider seam: real Claude calls, or a deterministic offline stub.

`get_client()` is what every entry point should call -- it decides between
`AnthropicClient` and `StubClient` and logs one clear line when it falls
back, so nobody is surprised about which one answered.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

import anthropic

from .config import Config
from .errors import LLMCallError

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """The three model calls this starter's pipeline needs."""

    def summarize(self, source_name: str, content: str) -> str: ...

    def extract_facts(self, source_name: str, content: str) -> list[str]:
        """Return up to a handful of standalone factual statements."""
        ...

    def synthesize(self, per_source_results: list[dict]) -> str:
        """Produce a roll-up synthesis across every processed source."""
        ...


def _text_from_response(resp: anthropic.types.Message) -> str:
    parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
    return "".join(parts).strip()


class AnthropicClient:
    """Real implementation, backed by the Claude API."""

    def __init__(self, config: Config) -> None:
        self._client = anthropic.Anthropic()
        self._model = config.anthropic_model

    def _create(self, **kwargs: object) -> anthropic.types.Message:
        try:
            return self._client.messages.create(model=self._model, **kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMCallError(f"authentication failed: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMCallError(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMCallError(f"API error (status {exc.status_code}): {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMCallError(f"connection error: {exc}") from exc

    def summarize(self, source_name: str, content: str) -> str:
        resp = self._create(
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following source, titled {source_name!r}, "
                        f"in 2-3 sentences:\n\n{content}"
                    ),
                }
            ],
        )
        return _text_from_response(resp)

    def extract_facts(self, source_name: str, content: str) -> list[str]:
        resp = self._create(
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Extract up to 5 standalone factual statements from this source, "
                        f"titled {source_name!r}:\n\n{content}"
                    ),
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "facts": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["facts"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        data = json.loads(_text_from_response(resp))
        return list(data["facts"])[:5]

    def synthesize(self, per_source_results: list[dict]) -> str:
        summary_block = "\n\n".join(f"- {r['source']}: {r['summary']}" for r in per_source_results)
        resp = self._create(
            max_tokens=2000,
            thinking={"type": "adaptive"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write a roll-up synthesis across these {len(per_source_results)} "
                        f"sources. Call out agreements, contradictions, and gaps.\n\n"
                        f"{summary_block}"
                    ),
                }
            ],
        )
        return _text_from_response(resp)


class StubClient:
    """Deterministic, offline, no network. Derives real output from real input.

    This is a stub for offline demos and tests -- not a stand-in for model
    quality. It never invents content: summaries are extractive, and "facts"
    are sentences pulled straight from the source text.
    """

    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def summarize(self, source_name: str, content: str) -> str:
        first_sentence = content.strip().split(".")[0].strip()
        word_count = len(content.split())
        return f"[stub] {source_name}: {first_sentence}. ({word_count} words total)"

    def extract_facts(self, source_name: str, content: str) -> list[str]:
        sentences = [s.strip() for s in self._SENTENCE_SPLIT.split(content) if s.strip()]
        # Prefer sentences that look information-bearing: contain a digit or a
        # capitalized proper-noun-like token.
        scored = [
            s for s in sentences if any(ch.isdigit() for ch in s) or re.search(r"[A-Z][a-z]+", s)
        ]
        facts = (scored or sentences)[:3]
        return [f"[stub] {source_name}: {fact}" for fact in facts]

    def synthesize(self, per_source_results: list[dict]) -> str:
        lines = [f"[stub] Synthesized {len(per_source_results)} source(s)."]
        for r in per_source_results:
            fact_count = len(r.get("facts", []))
            summary = r.get("summary", "")[:80]
            lines.append(f"- {r['source']}: {fact_count} fact(s) extracted; summary: {summary}")
        return "\n".join(lines)


def get_client(config: Config, *, force_offline: bool = False) -> LLMClient:
    """Pick StubClient or AnthropicClient per LLM_PROVIDER / --offline / key presence."""
    if force_offline or config.llm_provider == "stub":
        if not force_offline:
            logger.info(
                "LLM_PROVIDER=stub -- using the offline StubClient", extra={"provider": "stub"}
            )
        return StubClient()

    if config.llm_provider == "anthropic" and not config.anthropic_api_key:
        logger.info(
            "ANTHROPIC_API_KEY is not set -- falling back to the offline StubClient. "
            "Set ANTHROPIC_API_KEY (see .env.example) to use the real model.",
            extra={"provider": "stub", "reason": "missing_api_key"},
        )
        return StubClient()

    return AnthropicClient(config)
