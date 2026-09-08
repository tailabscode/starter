"""Model-provider seam: swap between the real Claude client and a
deterministic offline stub without touching any calling code.

get_client() defaults to AnthropicClient when ANTHROPIC_API_KEY is set,
falls back to StubClient otherwise (logging why), and --offline always
forces the stub. Tests use StubClient exclusively and never touch the
network.
"""

from __future__ import annotations

import logging
from typing import Protocol

import anthropic

from .config import Config
from .generation import SYSTEM_PROMPT, ContextBlock, build_user_prompt

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate_answer(self, query: str, blocks: list[ContextBlock]) -> str: ...


class AnthropicClient:
    """Real Claude client."""

    def __init__(self, model: str) -> None:
        self._client = anthropic.Anthropic()
        self._model = model

    def generate_answer(self, query: str, blocks: list[ContextBlock]) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(query, blocks)}],
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                "Anthropic API rejected the API key. Check ANTHROPIC_API_KEY in .env."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise RuntimeError("Anthropic API rate limit hit. Wait and retry.") from exc
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"Anthropic API returned an error: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"Could not reach the Anthropic API: {exc}") from exc
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )


class StubClient:
    """Deterministic, offline stand-in for AnthropicClient.

    Builds a real, useful answer from the actual retrieved chunks -- not a
    lorem-ipsum placeholder -- by stitching together a truncated snippet of
    each context block with its citation marker. It is honest string
    manipulation, not a language model; used for --offline runs and tests.
    """

    SNIPPET_CHARS = 220

    def generate_answer(self, query: str, blocks: list[ContextBlock]) -> str:
        if not blocks:
            return "[stub] I don't know -- no context was retrieved for this question."
        lines = [f"[stub] Grounded answer for: {query!r}"]
        for block in blocks:
            snippet = " ".join(block.chunk.text.split())[: self.SNIPPET_CHARS]
            lines.append(f"{snippet}... [{block.number}]")
        return " ".join(lines)


def get_client(config: Config, offline: bool) -> LLMClient:
    """Pick StubClient or AnthropicClient per --offline / LLM_PROVIDER / API key."""
    if offline:
        return StubClient()

    provider = config.llm_provider
    if provider == "stub":
        return StubClient()
    if provider == "anthropic":
        if not config.anthropic_api_key:
            logger.info(
                "ANTHROPIC_API_KEY not set; falling back to StubClient. "
                "Set ANTHROPIC_API_KEY in .env, or LLM_PROVIDER=stub to silence this."
            )
            return StubClient()
        return AnthropicClient(model=config.anthropic_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'anthropic' or 'stub')")
