"""Coordinator subtask bounds: never more than max_subtasks, and always at least one."""

from multi_agent_research.coordinator import decompose
from multi_agent_research.corpus import Corpus, Document
from multi_agent_research.llm import StubClient
from multi_agent_research.schemas import SubtaskPlan


def test_decompose_respects_max_subtasks(corpus) -> None:
    plan = decompose("How risky is EV battery supply?", corpus, StubClient(), max_subtasks=2)
    assert 0 < len(plan.subtasks) <= 2


def test_decompose_never_exceeds_stated_maximum_even_with_more_topics(corpus) -> None:
    # The bundled corpus has 5 topic documents; ask for at most 1.
    plan = decompose("What drives EV battery costs?", corpus, StubClient(), max_subtasks=1)
    assert len(plan.subtasks) == 1


def test_decompose_subtask_ids_are_unique(corpus) -> None:
    plan = decompose("How risky is EV battery supply?", corpus, StubClient(), max_subtasks=4)
    ids = [s.id for s in plan.subtasks]
    assert len(ids) == len(set(ids))


def test_decompose_falls_back_when_no_subtasks_proposed() -> None:
    """Even if the model (or a misbehaving client) proposes zero subtasks, one is guaranteed."""

    class _EmptyPlanClient(StubClient):
        def structured(self, *, system, user_content, schema, thinking=True):  # noqa: ANN001
            del system, user_content, thinking
            return SubtaskPlan(reasoning="nothing proposed", subtasks=[])

    doc = Document(doc_id="only", source="only.md", title="Only", text="# Only\ntext")
    one_doc_corpus = Corpus([doc])
    plan = decompose("A question", one_doc_corpus, _EmptyPlanClient(), max_subtasks=4)

    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].id == "full-question"
