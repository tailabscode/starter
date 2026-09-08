"""Environment-driven configuration, loaded once via a frozen dataclass."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PACKAGE_ROOT = Path(__file__).resolve().parent
_STARTER_ROOT = _PACKAGE_ROOT.parent.parent
DEFAULT_DATA_DIR = _STARTER_ROOT / "data"
DEFAULT_TRACE_PATH = Path(tempfile.gettempdir()) / "agentic_rag_last_trace.json"


@dataclass(frozen=True)
class Config:
    """Runtime configuration resolved from environment variables."""

    anthropic_api_key: str | None
    model: str
    llm_provider: str
    voyage_api_key: str | None
    data_dir: Path
    trace_path: Path
    max_steps: int
    log_level: str


def load_config() -> Config:
    """Build a :class:`Config` from the current environment.

    Reads the environment fresh on every call (rather than caching a module-level singleton)
    so tests can monkeypatch ``os.environ`` and see the effect immediately.
    """
    data_dir = Path(os.environ.get("AGENTIC_RAG_DATA_DIR", str(DEFAULT_DATA_DIR)))
    trace_path = Path(os.environ.get("AGENTIC_RAG_TRACE_PATH", str(DEFAULT_TRACE_PATH)))
    return Config(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
        llm_provider=os.environ.get("LLM_PROVIDER", "").strip().lower(),
        voyage_api_key=os.environ.get("VOYAGE_API_KEY") or None,
        data_dir=data_dir,
        trace_path=trace_path,
        max_steps=int(os.environ.get("AGENTIC_RAG_MAX_STEPS", "6")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
