"""State machine: legal transitions succeed, illegal ones raise."""

import pytest

from long_running_agent.errors import InvalidTransitionError
from long_running_agent.models import JobStatus, validate_transition


@pytest.mark.parametrize(
    ("frm", "to"),
    [
        (JobStatus.PENDING, JobStatus.RUNNING),
        (JobStatus.PENDING, JobStatus.CANCELLED),
        (JobStatus.RUNNING, JobStatus.CHECKPOINTED),
        (JobStatus.RUNNING, JobStatus.FAILED),
        (JobStatus.RUNNING, JobStatus.CANCELLED),
        (JobStatus.CHECKPOINTED, JobStatus.RUNNING),
        (JobStatus.CHECKPOINTED, JobStatus.COMPLETED),
        (JobStatus.CHECKPOINTED, JobStatus.CANCELLED),
    ],
)
def test_legal_transitions_are_allowed(frm: JobStatus, to: JobStatus) -> None:
    validate_transition(frm, to)  # must not raise


@pytest.mark.parametrize(
    ("frm", "to"),
    [
        (JobStatus.COMPLETED, JobStatus.RUNNING),
        (JobStatus.PENDING, JobStatus.COMPLETED),
        (JobStatus.FAILED, JobStatus.RUNNING),
        (JobStatus.CANCELLED, JobStatus.RUNNING),
        (JobStatus.COMPLETED, JobStatus.PENDING),
        (JobStatus.CHECKPOINTED, JobStatus.PENDING),
        (JobStatus.PENDING, JobStatus.PENDING),
        (JobStatus.COMPLETED, JobStatus.COMPLETED),
        (JobStatus.RUNNING, JobStatus.COMPLETED),
    ],
)
def test_illegal_transitions_raise(frm: JobStatus, to: JobStatus) -> None:
    with pytest.raises(InvalidTransitionError):
        validate_transition(frm, to)


def test_terminal_states_have_no_outgoing_edges() -> None:
    from long_running_agent.models import VALID_TRANSITIONS

    for terminal in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        assert VALID_TRANSITIONS[terminal] == frozenset()
