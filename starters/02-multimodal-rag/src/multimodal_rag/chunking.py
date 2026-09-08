"""Structure-aware markdown/text chunking.

Splits on markdown ATX headings first (so each chunk carries the heading
trail it came from), then applies a token-ish word-count budget with
overlap inside each heading section. Produces the same `Item` type as
image ingestion (see image_ingest.py) so text chunks and image captions
land in one unified list to index. See 01-basic-rag for the same chunker
in isolation; duplicated here on purpose, per this library's "standalone
starter" design.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .items import Item

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

CHUNK_WORDS = 180
OVERLAP_WORDS = 40


@dataclass(frozen=True)
class _Section:
    heading_trail: list[str]
    text: str


def _split_sections(raw_text: str) -> list[_Section]:
    """Split raw markdown text into sections keyed by their heading trail."""
    stack: list[tuple[int, str]] = []  # (heading level, title)
    sections: list[_Section] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(_Section(heading_trail=[title for _, title in stack], text=text))
        buffer.clear()

    for line in raw_text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buffer.append(line)
    flush()

    if not sections:
        stripped = raw_text.strip()
        if stripped:
            sections.append(_Section(heading_trail=[], text=stripped))
    return sections


def _word_budget_chunks(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Split text into overlapping windows of at most chunk_words words each."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    step = chunk_words - overlap_words
    pieces: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_words]
        pieces.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
        start += step
    return pieces


def _stable_item_id(source: str, heading_trail: list[str], chunk_index: int, text: str) -> str:
    """A short, deterministic id derived from where a chunk came from and its content."""
    digest = hashlib.sha1(
        f"{source}::{'>'.join(heading_trail)}::{chunk_index}::{text[:80]}".encode()
    ).hexdigest()
    return digest[:12]


def chunk_document(
    source: str,
    raw_text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Item]:
    """Chunk one document's raw text into structure-aware, overlapping text Items."""
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be >= 0 and < chunk_words")

    items: list[Item] = []
    index = 0
    for section in _split_sections(raw_text):
        for piece in _word_budget_chunks(section.text, chunk_words, overlap_words):
            item_id = _stable_item_id(source, section.heading_trail, index, piece)
            items.append(
                Item(
                    item_id=item_id,
                    kind="text",
                    source=source,
                    text=piece,
                    heading_trail=section.heading_trail,
                )
            )
            index += 1
    return items


def chunk_corpus(
    corpus_dir: Path,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Item]:
    """Chunk every .md/.txt file directly inside corpus_dir (sorted for determinism)."""
    paths = sorted(
        p for p in corpus_dir.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    )
    items: list[Item] = []
    for path in paths:
        raw_text = path.read_text(encoding="utf-8")
        items.extend(chunk_document(path.name, raw_text, chunk_words, overlap_words))
    return items
