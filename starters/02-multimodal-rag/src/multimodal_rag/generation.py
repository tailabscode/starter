"""Grounded multimodal generation.

Retrieved items are numbered into content blocks: text chunks become a
single labeled text block; image items become a labeled text block
*followed by a real image content block built from the actual file on
disk* -- the model answers from the pixels, not from the caption that got
it retrieved. The model's [n] citation markers are then resolved back to
real item ids and source files, the same way as 01-basic-rag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .images import build_image_content_block
from .items import Item

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions using only the numbered "
    "context blocks provided below, some of which are images. Cite every claim "
    "with the matching bracketed number, e.g. [1] or [2]. When an image block is "
    "relevant, describe what you actually see in it rather than repeating its "
    "caption verbatim. If the context does not contain enough information to "
    "answer, say plainly that you don't know -- do not guess or rely on outside "
    "knowledge."
)


@dataclass(frozen=True)
class ContentItem:
    """A retrieved item, numbered for citation in the prompt and the answer."""

    number: int
    item: Item


@dataclass(frozen=True)
class Citation:
    marker: int
    item_id: str | None
    source: str | None
    resolved: bool


@dataclass(frozen=True)
class AnswerResult:
    text: str
    citations: list[Citation]
    unresolved_markers: list[int]


def build_content_items(items: list[Item]) -> list[ContentItem]:
    """Number retrieved items 1..n in retrieval order for citation."""
    return [ContentItem(number=i + 1, item=item) for i, item in enumerate(items)]


def build_content_blocks(content_items: list[ContentItem]) -> list[dict]:
    """Build the Anthropic message content list from numbered items."""
    blocks: list[dict] = []
    for ci in content_items:
        item = ci.item
        if item.kind == "text":
            trail = " > ".join(item.heading_trail) or "(no heading)"
            blocks.append(
                {
                    "type": "text",
                    "text": f'[{ci.number}] source={item.source} heading="{trail}"\n{item.text}',
                }
            )
        elif item.kind == "image":
            if item.image_path is None or item.media_type is None:
                raise ValueError(f"Image item {item.item_id!r} is missing image_path/media_type")
            blocks.append({"type": "text", "text": f"[{ci.number}] source={item.source} (image)"})
            image_bytes = Path(item.image_path).read_bytes()
            blocks.append(build_image_content_block(image_bytes, item.media_type))
        else:
            raise ValueError(f"Unknown item kind: {item.kind!r}")
    return blocks


def resolve_citations(answer_text: str, content_items: list[ContentItem]) -> list[Citation]:
    """Map every [n] marker in the answer back to a real item, in first-seen order."""
    by_number = {ci.number: ci for ci in content_items}
    seen: set[int] = set()
    citations: list[Citation] = []
    for match in _CITATION_RE.finditer(answer_text):
        marker = int(match.group(1))
        if marker in seen:
            continue
        seen.add(marker)
        ci = by_number.get(marker)
        if ci is not None:
            citations.append(
                Citation(
                    marker=marker, item_id=ci.item.item_id, source=ci.item.source, resolved=True
                )
            )
        else:
            citations.append(Citation(marker=marker, item_id=None, source=None, resolved=False))
    return citations


def make_answer_result(answer_text: str, content_items: list[ContentItem]) -> AnswerResult:
    citations = resolve_citations(answer_text, content_items)
    unresolved = [c.marker for c in citations if not c.resolved]
    return AnswerResult(text=answer_text, citations=citations, unresolved_markers=unresolved)
