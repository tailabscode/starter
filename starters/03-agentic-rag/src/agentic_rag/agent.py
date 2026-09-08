"""The bounded agentic loop: plan once, then let the model decide whether/how to retrieve.

The plan's classification is not advisory -- it changes what actually happens next. A
``NO_RETRIEVAL`` plan skips the corpus entirely (the model is called with an empty tool list,
so it is structurally incapable of calling a tool). Anything else enters a loop bounded by
``max_steps``, which the model exits by returning a final text answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agentic_rag.llm import LLMClient
from agentic_rag.retrieval import HybridRetriever
from agentic_rag.schemas import QueryPlan, RunTrace
from agentic_rag.tools import TOOL_DEFS, execute_tool

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]+::\d+)\]")

_AGENT_SYSTEM = (
    "You are a retrieval agent. Use search_corpus to find evidence, list_topics if you are "
    "unsure whether the corpus covers the topic at all, and fetch_chunk to pull extra context "
    "around a promising hit. For a question that needs two hops, let what the first search "
    "returns inform your second search query -- do not just guess a second query up front. "
    "When you have enough evidence, answer in plain text and cite every claim with the exact "
    "chunk id it came from, in the form [chunk_id]. Only cite chunk ids you actually retrieved."
)

_NO_RETRIEVAL_SYSTEM = (
    "Answer the user's question directly from general knowledge. Do not claim to have "
    "consulted any document or corpus."
)


@dataclass
class RunResult:
    answer: str
    citations: list[str]
    trace: RunTrace
    plan: QueryPlan = field(repr=False)


def _build_task_prompt(question: str, plan: QueryPlan) -> str:
    return (
        f"Question: {question}\n\n"
        f"Planning notes (classification={plan.classification}): {plan.reasoning}\n"
        f"Suggested sub-queries (JSON): {json.dumps(plan.sub_queries)}\n\n"
        "Use the available tools to find grounded evidence, then answer with citations in the "
        "form [chunk_id] using only chunk ids you actually retrieved."
    )


def resolve_citations(text: str, retrieved_ids: set[str], valid_ids: set[str]) -> list[str]:
    """Keep only bracket citations that point at chunk ids we actually retrieved this run.

    A citation to a real corpus chunk that was never retrieved -- or to an id that does not
    exist at all -- is dropped rather than trusted, so the answer can never claim grounding it
    does not have.
    """
    found = _CITATION_RE.findall(text)
    resolved = [cid for cid in found if cid in retrieved_ids and cid in valid_ids]
    seen: list[str] = []
    for cid in resolved:
        if cid not in seen:
            seen.append(cid)
    return seen


def run_agent(
    question: str,
    *,
    llm: LLMClient,
    retriever: HybridRetriever,
    max_steps: int = 6,
) -> RunResult:
    """Run the plan-then-act loop for one question. Returns the answer plus a full trace."""
    trace = RunTrace(query=question)
    plan = llm.plan_query(question, retriever.list_topics())
    trace.plan = plan.model_dump()
    trace.add_step("plan", {"classification": plan.classification, "sub_queries": plan.sub_queries})

    if plan.classification == "NO_RETRIEVAL":
        result = llm.agent_step(
            system=_NO_RETRIEVAL_SYSTEM, messages=[{"role": "user", "content": question}], tools=[]
        )
        trace.add_step("final", {"tool_calls": 0})
        trace.stopped_reason = "no_retrieval"
        trace.answer = result.text
        trace.citations = []
        return RunResult(answer=result.text, citations=[], trace=trace, plan=plan)

    messages: list[dict] = [{"role": "user", "content": _build_task_prompt(question, plan)}]
    retrieved_ids: set[str] = set()
    valid_ids = retriever.valid_chunk_ids()
    final_answer = ""

    for step_number in range(1, max_steps + 1):
        result = llm.agent_step(system=_AGENT_SYSTEM, messages=messages, tools=TOOL_DEFS)

        if not result.tool_calls:
            trace.add_step("final", {"step": step_number, "text_preview": result.text[:200]})
            trace.stopped_reason = "end_turn"
            final_answer = result.text
            break

        messages.append({"role": "assistant", "content": result.content})
        tool_results = []
        for call in result.tool_calls:
            output, is_error = execute_tool(call.name, call.input, retriever)
            if call.name in ("search_corpus", "fetch_chunk") and not is_error:
                retrieved_ids.update(_chunk_ids_from_output(call.name, output))
            trace.add_step(
                "tool_call",
                {"step": step_number, "name": call.name, "input": call.input, "is_error": is_error},
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(output),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})
    else:
        trace.add_step("bound_hit", {"max_steps": max_steps})
        trace.stopped_reason = "max_steps"
        final_answer = (
            f"(stopped after {max_steps} steps without a final answer) "
            "Evidence gathered so far may be incomplete."
        )

    citations = resolve_citations(final_answer, retrieved_ids, valid_ids)
    trace.answer = final_answer
    trace.citations = citations
    return RunResult(answer=final_answer, citations=citations, trace=trace, plan=plan)


def _chunk_ids_from_output(tool_name: str, output: dict) -> set[str]:
    if tool_name == "search_corpus":
        return {hit["chunk_id"] for hit in output.get("results", [])}
    if tool_name == "fetch_chunk":
        ids = {output["chunk"]["chunk_id"]} if "chunk" in output else set()
        ids.update(c["chunk_id"] for c in output.get("context", []))
        return ids
    return set()
