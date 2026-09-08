"""Named exceptions so failures surface as clear messages, not bare tracebacks."""

from __future__ import annotations


class MissingCredentialsError(RuntimeError):
    """Raised when the Anthropic provider is requested but no API key is configured."""


class UnknownChunkError(KeyError):
    """Raised when a tool call references a chunk id that does not exist in the corpus."""


class CorpusNotFoundError(FileNotFoundError):
    """Raised when the configured corpus data directory has no markdown documents."""
