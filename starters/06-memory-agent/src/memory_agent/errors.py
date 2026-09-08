"""Named exceptions. Nothing here is a bare KeyError/AuthenticationError leak."""


class MemoryAgentError(Exception):
    """Base class for every error this package raises on purpose."""


class MissingCredentialsError(MemoryAgentError):
    """Raised when a real API call is attempted without credentials configured."""


class LLMCallError(MemoryAgentError):
    """Wraps any failure talking to the model provider (network, auth, rate limit)."""


class MemoryNotFoundError(MemoryAgentError):
    """Raised when a memory id does not exist in the store."""
