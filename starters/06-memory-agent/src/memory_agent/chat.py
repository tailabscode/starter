"""One chat turn: retrieve relevant memories, respond, then extract and write new ones.

This is the one place that knows how session memory, persistent memory, and
the model fit together -- `cli.py` only prints what this returns.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMClient
from .models import MemoryRecord
from .session import SessionMemory
from .store import MemoryStore


@dataclass
class ChatTurnResult:
    reply: str
    retrieved: list[tuple[MemoryRecord, float]]
    memory_writes: list[tuple[MemoryRecord, str]]  # (record, "create" | "update")


def build_system_prompt(retrieved: list[tuple[MemoryRecord, float]]) -> str:
    """The system prompt's memory block -- clearly separate from the live conversation."""
    if not retrieved:
        remembered = "(nothing remembered yet about this user)"
    else:
        remembered = "\n".join(
            f"- ({record.category}) {record.content}" for record, _score in retrieved
        )
    return (
        "You are a helpful assistant with persistent memory about this user.\n\n"
        "## What you remember about this user\n"
        f"{remembered}\n\n"
        "The block above is durable memory from prior sessions. It is separate "
        "from the live conversation below -- treat it as background context, "
        "not something the user just said."
    )


def run_turn(
    session: SessionMemory,
    store: MemoryStore,
    llm_client: LLMClient,
    user_message: str,
    *,
    top_k: int,
    dedup_threshold: float,
    max_memories_per_turn: int,
) -> ChatTurnResult:
    """Run one full turn: retrieve -> respond -> extract -> bounded, deduplicated writes."""
    session.add("user", user_message)

    retrieved = store.retrieve_top_k(user_message, top_k)
    system_prompt = build_system_prompt(retrieved)
    reply = llm_client.respond(system_prompt, session.messages)
    session.add("assistant", reply)

    exchange = session.messages[-2:]  # just the turn that happened, not the whole conversation
    proposals = llm_client.extract_memories(exchange)[:max_memories_per_turn]

    memory_writes: list[tuple[MemoryRecord, str]] = []
    for proposal in proposals:
        record, action_taken = store.upsert_with_dedup(
            proposal["content"],
            proposal["category"],
            session.session_id,
            dedup_threshold,
        )
        memory_writes.append((record, action_taken))

    return ChatTurnResult(reply=reply, retrieved=retrieved, memory_writes=memory_writes)
