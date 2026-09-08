"""Named exceptions raised at the boundaries of this package.

Callers should never see a bare KeyError or anthropic.AuthenticationError
traceback -- CLI code catches these and prints a clear, actionable message.
"""

from __future__ import annotations


class MissingCredentialsError(Exception):
    """Raised when a required API key is absent and no offline path applies."""


class EmbedderMismatchError(Exception):
    """Raised when the requested mode can't use the embedder an index was built with."""
