from agentic_rag.retrieval import HybridRetriever
from agentic_rag.stub_logic import stub_plan


def test_no_retrieval_for_general_knowledge_question(retriever: HybridRetriever) -> None:
    plan = stub_plan("What is 2 + 2?", retriever.list_topics())
    assert plan.classification == "NO_RETRIEVAL"
    assert plan.sub_queries == []


def test_single_lookup_for_one_hop_question(retriever: HybridRetriever) -> None:
    plan = stub_plan("What is Nimbus?", retriever.list_topics())
    assert plan.classification == "SINGLE_LOOKUP"
    assert len(plan.sub_queries) == 1


def test_multi_hop_for_chained_question(retriever: HybridRetriever) -> None:
    question = "Who rebuilt Nimbus's caching layer, and what other systems does that team own?"
    plan = stub_plan(question, retriever.list_topics())
    assert plan.classification == "MULTI_HOP"
    assert len(plan.sub_queries) == 2
