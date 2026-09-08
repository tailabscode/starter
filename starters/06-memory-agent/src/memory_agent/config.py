"""Frozen config loaded from the environment. ``load_dotenv()`` runs once, here."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str | None
    anthropic_model: str
    llm_provider: str
    log_level: str
    db_path: str
    embedding_dim: int
    retrieval_top_k: int
    dedup_threshold: float
    max_memories_per_turn: int


def load_config() -> Config:
    """Read configuration from the environment. Safe to call repeatedly."""
    return Config(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
        llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        db_path=os.environ.get("MEMORY_AGENT_DB", "memory_agent.db"),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "64")),
        retrieval_top_k=int(os.environ.get("RETRIEVAL_TOP_K", "3")),
        dedup_threshold=float(os.environ.get("DEDUP_THRESHOLD", "0.82")),
        max_memories_per_turn=int(os.environ.get("MAX_MEMORIES_PER_TURN", "3")),
    )
