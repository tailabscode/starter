"""long_running_agent: checkpointed, resumable, retryable background jobs.

The package demonstrates the "kick off a job, come back later, it survived
a crash" pattern: job state and step-level checkpoints live in SQLite, so a
job can be resumed from its last completed step instead of from scratch.
"""

__version__ = "0.1.0"
