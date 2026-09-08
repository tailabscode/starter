"""Job lifecycle enum + validated state machine, and the two row dataclasses.

Lifecycle: PENDING -> RUNNING -> (CHECKPOINTED per step, looping back to
RUNNING for the next step) -> COMPLETED | FAILED | CANCELLED.

CHECKPOINTED and RUNNING alternate once per step: every completed step is
durably persisted (status=CHECKPOINTED, checkpoint row written) before the
next step is attempted (status back to RUNNING). COMPLETED is only reached
from CHECKPOINTED, right after the final step's checkpoint has been written.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import InvalidTransitionError


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Terminal states (COMPLETED, FAILED, CANCELLED) have no outgoing edges.
VALID_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.PENDING: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset({JobStatus.CHECKPOINTED, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.CHECKPOINTED: frozenset(
        {JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.CANCELLED}
    ),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


def validate_transition(frm: JobStatus, to: JobStatus) -> None:
    """Raise InvalidTransitionError unless `frm -> to` is an allowed edge.

    A status "changing" to itself is never valid here (e.g. RUNNING -> RUNNING
    is handled as an explicit idempotent no-op by callers that need it, such as
    resuming a job a crash left mid-step -- it is not a state machine edge).
    """
    allowed = VALID_TRANSITIONS.get(frm, frozenset())
    if to not in allowed:
        raise InvalidTransitionError(f"illegal job transition: {frm.value} -> {to.value}")


@dataclass(frozen=True)
class Job:
    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    input: str  # JSON-encoded list[dict] of sources
    current_step: int
    total_steps: int
    result: str | None  # JSON-encoded dict once COMPLETED
    error: str | None


@dataclass(frozen=True)
class Checkpoint:
    id: int
    job_id: str
    step_index: int
    step_name: str
    payload_json: str
    created_at: str
