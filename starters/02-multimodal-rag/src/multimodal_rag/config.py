"""Frozen application config, built once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of the environment this process runs with."""

    anthropic_api_key: str | None
    anthropic_model: str
    voyage_api_key: str | None
    llm_provider: str
    log_level: str
    top_k: int
    rrf_k: int

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
            voyage_api_key=os.environ.get("VOYAGE_API_KEY") or None,
            llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            top_k=int(os.environ.get("TOP_K", "5")),
            rrf_k=int(os.environ.get("RRF_K", "60")),
        )
