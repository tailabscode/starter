"""Deterministic, offline heuristics behind :class:`multi_agent_research.llm.StubClient`.

None of this is a language model. Each function is a small, testable rule over structured data
(a question, a topic list, a findings list); the ``stub_*`` wrappers exist only to parse that
same data back out of the prompt strings the real API would also receive, so both clients
satisfy the exact same ``LLMClient`` interface. See the README's "Limitations" section.
"""

from __future__ import annotations

import json
import re

from multi_agent_research.schemas import (
    AgentStepResult,
    Critique,
    Subtask,
    SubtaskPlan,
    SynthesisReport,
    ToolCall,
)

_EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0}


def decompose_from_data(question: str, topics: list[dict], max_subtasks: int) -> SubtaskPlan:
    """One subtask per corpus topic, bounded to ``max_subtasks``.

    Deliberately topic-driven rather than clause-parsed: it ties each subtask directly to a
    real corpus document, which makes the demo's gap-then-revise flow concrete (see the
    README) -- if the corpus has more topics than ``max_subtasks`` allows, the leftover topic
    becomes exactly the gap the critic should catch.
    """
    chosen = topics[: max(1, max_subtasks)]
    subtasks = [
        Subtask(id=t["doc_id"], description=f"Research {t['title']} as it relates to: {question}")
        for t in chosen
    ]
    return SubtaskPlan(
        reasoning=(
            f"Split into one subtask per corpus topic, up to the max_subtasks bound of "
            f"{max_subtasks}."
        ),
        subtasks=subtasks,
    )


def critique_from_data(topics: list[dict], findings: list[dict]) -> Critique:
    """Sufficient iff every corpus topic is represented among the findings' sources."""
    covered = {src["doc_id"] for f in findings for src in f.get("sources", [])}
    target = {t["doc_id"]: t["title"] for t in topics}
    missing = [doc_id for doc_id in target if doc_id not in covered]
    if not missing:
        return Critique(is_sufficient=True, gaps=[], reasoning="Every corpus topic is covered.")
    gaps = [f"No findings cover: {target[doc_id]} ({doc_id})" for doc_id in missing]
    return Critique(
        is_sufficient=False,
        gaps=gaps,
        reasoning=f"{len(missing)} corpus topic(s) are not represented in any finding's sources.",
    )


def synthesize_from_data(question: str, findings: list[dict]) -> SynthesisReport:
    """Concatenate each finding's answer under its subtask heading, citing every source used.

    Not a real synthesis (no cross-finding reasoning) -- it demonstrates the same contract a
    real model must honor: every citation in the report traces back to a source a researcher
    actually retrieved, never an invented one.
    """
    if not findings:
        return SynthesisReport(
            report=f"(offline stub) No findings were collected for: {question}", citations=[]
        )
    sections = []
    citations: list[str] = []
    for finding in findings:
        doc_ids = [src["doc_id"] for src in finding.get("sources", [])]
        citations.extend(doc_ids)
        cite_suffix = "".join(f"[{doc_id}]" for doc_id in doc_ids)
        sections.append(f"### {finding['subtask_id']}\n{finding['answer']} {cite_suffix}".strip())
    seen: set[str] = set()
    ordered_citations = [c for c in citations if not (c in seen or seen.add(c))]
    report = f"(offline stub) Synthesis for: {question}\n\n" + "\n\n".join(sections)
    return SynthesisReport(report=report, citations=ordered_citations)


def _extract_json_field(prompt: str, label: str) -> object | None:
    match = re.search(rf"{re.escape(label)}: (\[.*\]|\{{.*\}})", prompt)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def stub_decompose(user_content: str) -> SubtaskPlan:
    question_match = re.search(r"Question: (.*?)\n\n", user_content, re.DOTALL)
    question = question_match.group(1).strip() if question_match else user_content.strip()
    topics = _extract_json_field(user_content, "Corpus topics (JSON)") or []
    max_match = re.search(r"Maximum subtasks: (\d+)", user_content)
    max_subtasks = int(max_match.group(1)) if max_match else 4
    return decompose_from_data(question, topics, max_subtasks)


def stub_critique(user_content: str) -> Critique:
    topics = _extract_json_field(user_content, "All corpus topics (JSON)") or []
    findings = _extract_json_field(user_content, "Findings so far (JSON)") or []
    return critique_from_data(topics, findings)


def stub_synthesize(user_content: str) -> SynthesisReport:
    question_match = re.search(r"Question: (.*?)\n\n", user_content, re.DOTALL)
    question = question_match.group(1).strip() if question_match else user_content.strip()
    findings = _extract_json_field(user_content, "Findings (JSON)") or []
    return synthesize_from_data(question, findings)


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


def _prior_search_count(messages: list[dict]) -> int:
    return sum(1 for b in _iter_blocks(messages, "tool_use") if b.get("name") == "search_corpus")


def _latest_search_results(messages: list[dict]) -> list[dict]:
    for block in reversed(_iter_blocks(messages, "tool_result")):
        try:
            payload = json.loads(block.get("content", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if "results" in payload:
            return payload["results"]
    return []


def stub_agent_step(messages: list[dict], tools: list[dict]) -> AgentStepResult:
    """A researcher's bounded tool loop: search once, then summarize what came back."""
    tool_names = {t["name"] for t in tools if "name" in t}

    if "search_corpus" in tool_names and _prior_search_count(messages) == 0:
        prompt = _first_user_text(messages)
        query = prompt.split("\n\n")[0].removeprefix("Subtask:").strip()
        call = ToolCall(id="stub_call_1", name="search_corpus", input={"query": query, "top_k": 2})
        block = {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
        return AgentStepResult(
            stop_reason="tool_use", content=[block], tool_calls=[call], text="", usage=_EMPTY_USAGE
        )

    hits = _latest_search_results(messages)
    if hits:
        fragments = [f"{h['title']}: {h['text'].strip().split('. ')[0].rstrip('.')}." for h in hits]
        text = f"(offline stub) {' '.join(fragments)}"
    else:
        text = "(offline stub) No relevant evidence was found in the corpus for this subtask."
    return AgentStepResult(
        stop_reason="end_turn",
        content=[{"type": "text", "text": text}],
        tool_calls=[],
        text=text,
        usage=_EMPTY_USAGE,
    )
