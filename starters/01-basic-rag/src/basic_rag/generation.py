"""Grounded answer generation: build numbered context blocks, instruct the
model to cite them as [1], [2]..., then resolve those markers back to real
chunk ids and source files. A marker the model produced that doesn't map
to any retrieved block is reported as unresolved, never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunking import Chunk

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions using only the numbered "
    "context blocks provided below. Cite every claim with the matching "
    "bracketed number, e.g. [1] or [2]. Use multiple citations if a claim "
    "draws on more than one block. If the context does not contain enough "
    "information to answer, say plainly that you don't know -- do not guess "
    "or rely on outside knowledge."
)


@dataclass(frozen=True)
class ContextBlock:
    """A retrieved chunk, numbered for citation in the prompt and the answer."""

    number: int
    chunk: Chunk


@dataclass(frozen=True)
class Citation:
    marker: int
    chunk_id: str | None
    source: str | None
    resolved: bool


@dataclass(frozen=True)
class AnswerResult:
    text: str
    citations: list[Citation]
    unresolved_markers: list[int]


def build_context_blocks(chunks: list[Chunk]) -> list[ContextBlock]:
    """Number retrieved chunks 1..n in retrieval order for citation."""
    return [ContextBlock(number=i + 1, chunk=chunk) for i, chunk in enumerate(chunks)]


def format_context(blocks: list[ContextBlock]) -> str:
    parts = []
    for block in blocks:
        trail = " > ".join(block.chunk.heading_trail) or "(no heading)"
        parts.append(
            f'[{block.number}] source={block.chunk.source} heading="{trail}"\n{block.chunk.text}'
        )
    return "\n\n".join(parts)


def build_user_prompt(query: str, blocks: list[ContextBlock]) -> str:
    return f"Context:\n{format_context(blocks)}\n\nQuestion: {query}"


def resolve_citations(answer_text: str, blocks: list[ContextBlock]) -> list[Citation]:
    """Map every [n] marker in the answer back to a real chunk, in first-seen order."""
    by_number = {block.number: block for block in blocks}
    seen: set[int] = set()
    citations: list[Citation] = []
    for match in _CITATION_RE.finditer(answer_text):
        marker = int(match.group(1))
        if marker in seen:
            continue
        seen.add(marker)
        block = by_number.get(marker)
        if block is not None:
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=block.chunk.chunk_id,
                    source=block.chunk.source,
                    resolved=True,
                )
            )
        else:
            citations.append(Citation(marker=marker, chunk_id=None, source=None, resolved=False))
    return citations


def make_answer_result(answer_text: str, blocks: list[ContextBlock]) -> AnswerResult:
    citations = resolve_citations(answer_text, blocks)
    unresolved = [c.marker for c in citations if not c.resolved]
    return AnswerResult(text=answer_text, citations=citations, unresolved_markers=unresolved)
