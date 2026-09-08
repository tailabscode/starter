"""Named exceptions so failures surface as clear messages, not bare tracebacks."""

from __future__ import annotations


class MissingCredentialsError(RuntimeError):
    """Raised when the Anthropic provider is requested but no API key is configured."""


class RunNotFoundError(FileNotFoundError):
    """Raised when `show <run-id>` is given a run id with no matching state file."""


class CorpusNotFoundError(FileNotFoundError):
    """Raised when the configured corpus data directory has no markdown documents."""
