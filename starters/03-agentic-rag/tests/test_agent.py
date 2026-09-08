from agentic_rag.agent import resolve_citations, run_agent
from agentic_rag.llm import StubClient
from agentic_rag.retrieval import HybridRetriever
from agentic_rag.schemas import AgentStepResult, QueryPlan, ToolCall

MULTI_HOP_QUESTION = (
    "Who rebuilt Nimbus's caching layer, and what other systems does that team own?"
)


class InfiniteToolLLM:
    """A fake LLMClient that always requests another tool call, to exercise the max_steps bound."""

    def __init__(self) -> None:
        self.calls = 0

    def plan_query(self, question: str, topics: list[dict]) -> QueryPlan:
        return QueryPlan(classification="SINGLE_LOOKUP", reasoning="test", sub_queries=[question])

    def agent_step(self, system: str, messages: list[dict], tools: list[dict]) -> AgentStepResult:
        del system, messages, tools
        self.calls += 1
        call = ToolCall(
            id=f"call_{self.calls}", name="search_corpus", input={"query": "Aurora", "top_k": 1}
        )
        block = {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
        return AgentStepResult(stop_reason="tool_use", content=[block], tool_calls=[call], text="")


def test_no_retrieval_path_makes_zero_tool_calls(retriever: HybridRetriever) -> None:
    result = run_agent("What is 2 + 2?", llm=StubClient(), retriever=retriever, max_steps=6)
    assert result.trace.stopped_reason == "no_retrieval"
    tool_steps = [s for s in result.trace.steps if s.kind == "tool_call"]
    assert tool_steps == []
    assert result.citations == []


def test_multi_hop_path_issues_more_than_one_search(retriever: HybridRetriever) -> None:
    result = run_agent(MULTI_HOP_QUESTION, llm=StubClient(), retriever=retriever, max_steps=6)
    search_steps = [
        s
        for s in result.trace.steps
        if s.kind == "tool_call" and s.detail["name"] == "search_corpus"
    ]
    assert len(search_steps) > 1
    assert result.trace.stopped_reason == "end_turn"


def test_stub_answer_is_grounded_with_real_citations(retriever: HybridRetriever) -> None:
    result = run_agent(MULTI_HOP_QUESTION, llm=StubClient(), retriever=retriever, max_steps=6)
    assert result.citations
    for chunk_id in result.citations:
        assert chunk_id in retriever.valid_chunk_ids()


def test_max_steps_bound_is_enforced_and_reported(retriever: HybridRetriever) -> None:
    fake = InfiniteToolLLM()
    result = run_agent("Aurora team", llm=fake, retriever=retriever, max_steps=3)
    assert result.trace.stopped_reason == "max_steps"
    assert fake.calls == 3
    assert "3 steps" in result.answer
    bound_steps = [s for s in result.trace.steps if s.kind == "bound_hit"]
    assert len(bound_steps) == 1


def test_resolve_citations_refuses_invented_ids() -> None:
    retrieved = {"overview::0", "teams::0"}
    valid = retrieved | {"incidents::1"}
    text = "Aurora rebuilt the cache [overview::0]. It also owns other things [incidents::1]."
    resolved = resolve_citations(text, retrieved, valid)
    assert resolved == ["overview::0"]


def test_resolve_citations_deduplicates_and_preserves_order() -> None:
    retrieved = {"a::0", "b::0"}
    text = "First [a::0], then [b::0], then again [a::0]."
    resolved = resolve_citations(text, retrieved, retrieved)
    assert resolved == ["a::0", "b::0"]
