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

_MEMORY_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "update", "ignore"]},
                    "content": {"type": "string"},
                    "category": {"type": "string", "enum": ["preference", "fact", "episodic"]},
                },
                "required": ["action", "content", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}


class LLMClient(Protocol):
    """The two model calls this starter's chat loop needs."""

    def respond(self, system_prompt: str, messages: list[dict]) -> str:
        """Generate the assistant's reply for the current conversation."""
        ...

    def extract_memories(self, exchange: list[dict]) -> list[dict]:
        """Review one exchange and propose durable memory writes (create/update)."""
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

    def respond(self, system_prompt: str, messages: list[dict]) -> str:
        resp = self._create(
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        return _text_from_response(resp)

    def extract_memories(self, exchange: list[dict]) -> list[dict]:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in exchange)
        resp = self._create(
            max_tokens=500,
            output_config={
                "format": {"type": "json_schema", "schema": _MEMORY_PROPOSAL_SCHEMA},
                "effort": "low",
            },
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Review this exchange and propose durable memory writes about the "
                        "user: stable preferences, facts, or notable episodes worth "
                        "remembering in future conversations. Propose at most 3. Use "
                        "action 'update' only when the content clearly supersedes something "
                        "already known; otherwise use 'create'. If nothing is worth "
                        "remembering, return an empty list.\n\n"
                        f"{transcript}"
                    ),
                }
            ],
        )
        data = json.loads(_text_from_response(resp))
        return [m for m in data["memories"] if m["action"] != "ignore"]


class StubClient:
    """Deterministic, offline, no network. Derives real output from real input.

    This is a stub for offline demos and tests -- not a stand-in for model
    quality. `respond` only ever repeats what retrieval actually found (it
    cannot invent a memory), and `extract_memories` only proposes writes for
    sentences that literally match a first-person pattern ("I prefer ...",
    "I live ...", ...).
    """

    _REMEMBERED_LINE = re.compile(r"^- \([^)]+\)\s*(.+)$", re.MULTILINE)
    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
    _PREFERENCE_MARKERS = ("i prefer", "i like", "i love", "i hate", "i dislike", "i enjoy")
    _FACT_MARKERS = ("i live", "i work", "i am", "i'm", "my name is")

    def respond(self, system_prompt: str, messages: list[dict]) -> str:
        remembered = self._REMEMBERED_LINE.findall(system_prompt)
        if remembered:
            joined = "; ".join(remembered)
            return f"[stub] Based on what I remember about you: {joined}."
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f'[stub] Noted: "{last_user}" (nothing relevant remembered yet).'

    def extract_memories(self, exchange: list[dict]) -> list[dict]:
        proposals: list[dict] = []
        for message in exchange:
            if message["role"] != "user":
                continue
            for raw_sentence in self._SENTENCE_SPLIT.split(message["content"]):
                sentence = raw_sentence.strip()
                if not sentence:
                    continue
                # Anchored at the start on purpose: a declarative "I prefer X"
                # is worth remembering, but a question that merely mentions
                # "...do I prefer..." is not a statement about the user.
                lowered = sentence.lower()
                if any(lowered.startswith(marker) for marker in self._PREFERENCE_MARKERS):
                    category = "preference"
                elif any(lowered.startswith(marker) for marker in self._FACT_MARKERS):
                    category = "fact"
                else:
                    continue
                proposals.append(
                    {
                        "action": "create",
                        "content": _rephrase_first_person(sentence),
                        "category": category,
                    }
                )
        return proposals


def _rephrase_first_person(sentence: str) -> str:
    """Swap a leading first-person pronoun for "User", preserving the rest verbatim.

    Deliberately not grammar-aware (no conjugation) -- a real model call would
    do better, but preserving exact wording keeps the stub's output honestly
    plain, and keeps token overlap with later queries about the same fact.
    """
    if re.match(r"^i'm\b", sentence, re.IGNORECASE):
        return "User is" + sentence[3:]
    if re.match(r"^i am\b", sentence, re.IGNORECASE):
        return "User is" + sentence[4:]
    if re.match(r"^my\b", sentence, re.IGNORECASE):
        return "User's" + sentence[2:]
    if re.match(r"^i\b", sentence, re.IGNORECASE):
        return "User" + sentence[1:]
    return f"User said: {sentence}"


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
