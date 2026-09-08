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
    max_retries: int
    retry_base_delay_seconds: float
    max_wall_clock_seconds: float
    max_sources: int


def load_config() -> Config:
    """Read configuration from the environment. Safe to call repeatedly."""
    return Config(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
        llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        db_path=os.environ.get("LONG_RUNNING_AGENT_DB", "long_running_agent.db"),
        max_retries=int(os.environ.get("MAX_RETRIES", "3")),
        retry_base_delay_seconds=float(os.environ.get("RETRY_BASE_DELAY_SECONDS", "0.5")),
        max_wall_clock_seconds=float(os.environ.get("MAX_WALL_CLOCK_SECONDS", "300")),
        max_sources=int(os.environ.get("MAX_SOURCES", "25")),
    )
