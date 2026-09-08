"""Named exceptions. Nothing here is a bare KeyError/AuthenticationError leak."""


class LongRunningAgentError(Exception):
    """Base class for every error this package raises on purpose."""


class MissingCredentialsError(LongRunningAgentError):
    """Raised when a real API call is attempted without credentials configured."""


class InvalidTransitionError(LongRunningAgentError):
    """Raised when a job status transition is not allowed by the state machine."""


class JobNotFoundError(LongRunningAgentError):
    """Raised when a job id does not exist in the store."""


class TooManySourcesError(LongRunningAgentError):
    """Raised when a submission exceeds the configured MAX_SOURCES bound."""


class LLMCallError(LongRunningAgentError):
    """Wraps any failure talking to the model provider (network, auth, rate limit)."""


class RetryableStepError(LongRunningAgentError):
    """A step failure that is worth retrying with backoff."""


class SourceFetchError(RetryableStepError):
    """Simulated fetch of a source failed (transient, by design in this starter)."""


class StepFailedError(LongRunningAgentError):
    """A step exhausted its retry budget. Carries the last underlying error."""


class JobBoundExceededError(LongRunningAgentError):
    """A job hit its hard wall-clock ceiling. Prevents a runaway job looping forever."""
