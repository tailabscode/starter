"""Researchers: cheap workers that each run one subtask's bounded search-then-answer loop.

Researchers run concurrently via ``ThreadPoolExecutor`` -- the Anthropic client is thread-safe,
so each thread can safely call ``llm.agent_step`` on the same shared client instance. Each
researcher gets ``output_config={"effort": "low"}``: see the README for why cheap workers
feeding an expensive synthesiser is the right shape for this pipeline.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from multi_agent_research.corpus import Corpus
from multi_agent_research.llm import LLMClient, StubClient
from multi_agent_research.schemas import Subtask

logger = logging.getLogger("multi_agent_research")

_SYSTEM = (
    "You are a researcher on a team. You have been assigned one independent subtask. Use "
    "search_corpus to find evidence, then give a concise, well-supported answer to your "
    "subtask specifically -- do not try to answer the whole original question."
)

SEARCH_TOOL_DEF: dict = {
    "name": "search_corpus",
    "description": (
        "Search the local corpus for evidence relevant to a query. Returns matching documents "
        "with their doc id, source filename, title, and text, ranked by term overlap."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "top_k": {
                "type": "integer",
                "description": "Number of documents to return (1-5).",
                "minimum": 1,
                "maximum": 5,
            },
        },
        "required": ["query", "top_k"],
        "additionalProperties": False,
    },
}

_WEB_SEARCH_TOOL_DEF: dict = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}


def build_researcher_tools(*, enable_web_search: bool, is_stub: bool) -> list[dict]:
    """The search tool every researcher gets, plus the optional (off by default) web search.

    Web search is a paid, server-executed Anthropic tool -- see the README's cost note. It is
    skipped even when requested if the offline stub is in use, since a stub cannot execute a
    real network search and pretending otherwise would violate the "no network in tests" rule.
    """
    tools = [SEARCH_TOOL_DEF]
    if enable_web_search:
        if is_stub:
            logger.warning(
                "ENABLE_WEB_SEARCH is set but the offline stub is in use; web_search cannot "
                "run offline, so it was left out of this researcher's tools."
            )
        else:
            tools.append(_WEB_SEARCH_TOOL_DEF)
    return tools


def execute_search_tool(tool_input: dict, corpus: Corpus) -> tuple[dict, bool]:
    try:
        query = tool_input["query"]
        top_k = int(tool_input.get("top_k", 3))
        return {"results": corpus.search(query, top_k=top_k)}, False
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": f"Invalid tool input: {exc}"}, True


def research_subtask(
    subtask: Subtask,
    *,
    llm: LLMClient,
    corpus: Corpus,
    tools: list[dict],
    max_steps: int,
) -> dict:
    """Run one subtask's bounded tool loop. Returns a JSON-serializable finding."""
    messages: list[dict] = [
        {
            "role": "user",
            "content": f"Subtask: {subtask.description}\n\nFind evidence, then answer concisely.",
        }
    ]
    sources: dict[str, dict] = {}
    usage_total = {"input_tokens": 0, "output_tokens": 0}
    final_text = ""
    stopped_reason = "max_steps"

    for _step in range(1, max_steps + 1):
        result = llm.agent_step(system=_SYSTEM, messages=messages, tools=tools, effort="low")
        usage_total["input_tokens"] += result.usage.get("input_tokens", 0)
        usage_total["output_tokens"] += result.usage.get("output_tokens", 0)

        if not result.tool_calls:
            final_text = result.text
            stopped_reason = "end_turn"
            break

        messages.append({"role": "assistant", "content": result.content})
        tool_results = []
        for call in result.tool_calls:
            if call.name != "search_corpus":
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps({"error": f"Unknown tool: {call.name}"}),
                        "is_error": True,
                    }
                )
                continue
            output, is_error = execute_search_tool(call.input, corpus)
            if not is_error:
                for hit in output.get("results", []):
                    sources[hit["doc_id"]] = {
                        "doc_id": hit["doc_id"],
                        "source": hit["source"],
                        "title": hit["title"],
                    }
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
        final_text = f"(stopped after {max_steps} steps without a final answer for {subtask.id})"

    return {
        "subtask_id": subtask.id,
        "description": subtask.description,
        "answer": final_text,
        "sources": list(sources.values()),
        "usage": usage_total,
        "stopped_reason": stopped_reason,
    }


def run_researchers_concurrently(
    subtasks: list[Subtask],
    *,
    llm: LLMClient,
    corpus: Corpus,
    max_steps: int,
    enable_web_search: bool = False,
) -> list[dict]:
    """Dispatch every subtask to its own thread and collect all findings.

    The Anthropic client (and the stub) are safe to call from multiple threads at once, so this
    is real concurrency, not a sequential loop dressed up in executor syntax.
    """
    if not subtasks:
        return []
    tools = build_researcher_tools(
        enable_web_search=enable_web_search, is_stub=isinstance(llm, StubClient)
    )
    findings: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(subtasks)) as executor:
        future_to_subtask = {
            executor.submit(
                research_subtask, subtask, llm=llm, corpus=corpus, tools=tools, max_steps=max_steps
            ): subtask
            for subtask in subtasks
        }
        for future in as_completed(future_to_subtask):
            findings.append(future.result())
    findings.sort(key=lambda f: f["subtask_id"])
    return findings
