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
DEFAULT_RUNS_DIR = Path(tempfile.gettempdir()) / "multi_agent_research" / "runs"


@dataclass(frozen=True)
class Config:
    """Runtime configuration resolved from environment variables."""

    anthropic_api_key: str | None
    model: str
    llm_provider: str
    data_dir: Path
    runs_dir: Path
    max_subtasks: int
    max_rounds: int
    researcher_max_steps: int
    enable_web_search: bool
    log_level: str


def load_config() -> Config:
    """Build a :class:`Config` from the current environment.

    Reads the environment fresh on every call (rather than caching a module-level singleton)
    so tests can monkeypatch ``os.environ`` and see the effect immediately.
    """
    data_dir = Path(os.environ.get("MAR_DATA_DIR", str(DEFAULT_DATA_DIR)))
    runs_dir = Path(os.environ.get("MAR_RUNS_DIR", str(DEFAULT_RUNS_DIR)))
    return Config(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
        llm_provider=os.environ.get("LLM_PROVIDER", "").strip().lower(),
        data_dir=data_dir,
        runs_dir=runs_dir,
        max_subtasks=int(os.environ.get("MAR_MAX_SUBTASKS", "4")),
        max_rounds=int(os.environ.get("MAR_MAX_ROUNDS", "2")),
        researcher_max_steps=int(os.environ.get("MAR_RESEARCHER_MAX_STEPS", "3")),
        enable_web_search=os.environ.get("ENABLE_WEB_SEARCH", "false").strip().lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
