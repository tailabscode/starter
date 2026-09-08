"""Environment-driven configuration.

`load_dotenv()` runs once at import time so a local `.env` (copied from
`.env.example`) is picked up before any env var is read.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class MissingCredentialsError(RuntimeError):
    """Raised when a live API call is requested but no credential is configured."""


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str | None
    anthropic_model: str
    llm_provider: str  # "" (auto) | "anthropic" | "stub"
    log_level: str
    max_steps: int


def load_config() -> Config:
    """Build a `Config` from environment variables. Call once per process."""
    return Config(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
        llm_provider=os.environ.get("LLM_PROVIDER", "").strip().lower(),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        max_steps=int(os.environ.get("AGENT_MAX_STEPS", "8")),
    )
