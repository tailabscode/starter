"""Deterministic, offline heuristics behind :class:`agentic_rag.llm.StubClient`.

None of this is a language model. It is a small set of rules over the actual question text
and the actual retrieved chunk text, so the stub produces plausible, grounded output (real
chunk ids, real citations, a real second search informed by the first result set) without any
network access. It is honestly a heuristic, not a substitute for the real model -- see the
README's "Limitations" section.
"""

from __future__ import annotations

import json
import re

from agentic_rag.bm25 import tokenize
from agentic_rag.schemas import AgentStepResult, QueryPlan, ToolCall

_QUESTION_STARTERS = re.compile(r"^(what|who|which|where|when|how)\b")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z]+)\b")
_SKIP_ENTITIES = {"The", "Before", "After", "Current", "Any", "No"}


def stub_plan(question: str, topics: list[dict]) -> QueryPlan:
    """Classify a question by how much of it overlaps the corpus vocabulary."""
    vocab: set[str] = set()
    for topic in topics:
        vocab.update(tokenize(topic["title"]))
        vocab.add(topic["doc_id"].lower())
        for heading in topic["headings"]:
            vocab.update(tokenize(heading))

    question_tokens = tokenize(question)
    overlap = [t for t in question_tokens if t in vocab]

    if not overlap:
        return QueryPlan(
            classification="NO_RETRIEVAL",
            reasoning=(
                "None of the question's terms appear anywhere in the corpus vocabulary "
                f"({', '.join(t['doc_id'] for t in topics)}); this reads as general knowledge."
            ),
            sub_queries=[],
        )

    lower_q = question.lower()
    if " and " in lower_q:
        _before, after = lower_q.split(" and ", 1)
        if _QUESTION_STARTERS.match(after.strip()):
            parts = [p.strip(" ?.,") for p in re.split(r"\band\b", question, maxsplit=1)]
            sub_queries = [p for p in parts if p][:2]
            return QueryPlan(
                classification="MULTI_HOP",
                reasoning=(
                    "The question chains a lookup with a follow-up clause about an entity "
                    "that lookup should reveal, so it needs two searches in sequence."
                ),
                sub_queries=sub_queries or [question],
            )

    return QueryPlan(
        classification="SINGLE_LOOKUP",
        reasoning="The question maps to a single fact that one search should surface.",
        sub_queries=[question],
    )


def _first_user_text(messages: list[dict]) -> str:
    content = messages[0]["content"] if messages else ""
    return content if isinstance(content, str) else ""


def _iter_blocks(messages: list[dict], block_type: str) -> list[dict]:
    blocks = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        blocks.extend(b for b in content if isinstance(b, dict) and b.get("type") == block_type)
    return blocks


def _prior_tool_calls(messages: list[dict]) -> list[tuple[str, dict]]:
    return [(b["name"], b.get("input", {})) for b in _iter_blocks(messages, "tool_use")]


def _prior_tool_payloads(messages: list[dict]) -> list[dict]:
    payloads = []
    for block in _iter_blocks(messages, "tool_result"):
        try:
            payloads.append(json.loads(block.get("content", "")))
        except (json.JSONDecodeError, TypeError):
            continue
    return payloads


def _extract_sub_queries(prompt: str) -> list[str]:
    match = re.search(r"Suggested sub-queries \(JSON\): (\[.*\])", prompt)
    if not match:
        return []
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return []


def _body_only(hit: dict) -> str:
    """Drop a chunk's heading line so heading words (e.g. "Overview") aren't mistaken for
    entities -- the heading is structural, not a fact the text is pointing at."""
    return hit["text"].partition("\n")[2] or hit["text"]


def _extract_new_entity(question: str, hits: list[dict]) -> str | None:
    """Find a capitalized term in retrieved chunk bodies that was not already in the question.

    This is what makes the second hop "informed by the first result set" rather than a
    pre-planned guess: it looks at what the first search actually returned.
    """
    question_words = {w.lower() for w in re.findall(r"[A-Za-z]+", question)}
    for hit in hits:
        for candidate in _PROPER_NOUN_RE.findall(_body_only(hit)):
            if candidate in _SKIP_ENTITIES:
                continue
            if candidate.lower() not in question_words:
                return candidate
    return None


def _latest_search_hits(payloads: list[dict]) -> list[dict]:
    for payload in reversed(payloads):
        if "results" in payload:
            return payload["results"]
    return []


def _top_chunk_id(payloads: list[dict]) -> str | None:
    hits = _latest_search_hits(payloads)
    return hits[0]["chunk_id"] if hits else None


def _compose_answer(payloads: list[dict]) -> tuple[str, list[str]]:
    fragments: list[str] = []
    cited: list[str] = []
    for payload in payloads:
        hits = payload.get("results", [])
        for hit in hits[:1]:
            sentence = hit["text"].strip().split(". ")[0].rstrip(".")
            fragments.append(f"{sentence}. [{hit['chunk_id']}]")
            cited.append(hit["chunk_id"])
        if "chunk" in payload:
            hit = payload["chunk"]
            sentence = hit["text"].strip().split(". ")[0].rstrip(".")
            fragments.append(f"{sentence}. [{hit['chunk_id']}]")
            cited.append(hit["chunk_id"])
    body = " ".join(fragments) if fragments else "No relevant evidence was found in the corpus."
    return f"(offline stub) {body}", cited


def stub_agent_step(messages: list[dict], tools: list[dict]) -> AgentStepResult:
    """Deterministically decide the next tool call, or produce a grounded final answer."""
    tool_names = {t["name"] for t in tools}

    if not tool_names:
        question = _first_user_text(messages)
        text = (
            f"(offline stub) {question.strip()} is a general-knowledge question that does not "
            "require the corpus; answering directly without retrieval."
        )
        return AgentStepResult(
            stop_reason="end_turn",
            content=[{"type": "text", "text": text}],
            tool_calls=[],
            text=text,
        )

    prompt = _first_user_text(messages)
    question = prompt.split("\n\n")[0].removeprefix("Question:").strip()
    sub_queries = _extract_sub_queries(prompt)
    calls = _prior_tool_calls(messages)
    payloads = _prior_tool_payloads(messages)
    search_count = sum(1 for name, _ in calls if name == "search_corpus")
    fetch_count = sum(1 for name, _ in calls if name == "fetch_chunk")
    target_searches = max(1, min(2, len(sub_queries) or 1))

    if search_count < target_searches:
        if search_count == 0:
            query = sub_queries[0] if sub_queries else question
        else:
            hits = _latest_search_hits(payloads)
            entity = _extract_new_entity(question, hits)
            query = entity or (sub_queries[1] if len(sub_queries) > 1 else question)
        call = ToolCall(
            id=f"stub_call_{len(calls) + 1}",
            name="search_corpus",
            input={"query": query, "top_k": 3},
        )
        block = {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
        return AgentStepResult(stop_reason="tool_use", content=[block], tool_calls=[call], text="")

    if fetch_count < 1 and (chunk_id := _top_chunk_id(payloads)):
        call = ToolCall(
            id=f"stub_call_{len(calls) + 1}", name="fetch_chunk", input={"chunk_id": chunk_id}
        )
        block = {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
        return AgentStepResult(stop_reason="tool_use", content=[block], tool_calls=[call], text="")

    text, _cited = _compose_answer(payloads)
    return AgentStepResult(
        stop_reason="end_turn",
        content=[{"type": "text", "text": text}],
        tool_calls=[],
        text=text,
    )
