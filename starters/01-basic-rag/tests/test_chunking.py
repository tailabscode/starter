"""Tests for structure-aware chunking: heading trails and overlap boundaries."""

from __future__ import annotations

import pytest

from basic_rag.chunking import chunk_document


def _numbered_words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunks_carry_correct_heading_trail() -> None:
    text = (
        "# Title\n\n"
        "## Section A\n\n"
        f"{_numbered_words('a', 5)}\n\n"
        "## Section B\n\n"
        f"{_numbered_words('b', 5)}\n"
    )
    chunks = chunk_document("doc.md", text, chunk_words=100, overlap_words=10)

    assert len(chunks) == 2
    assert chunks[0].heading_trail == ["Title", "Section A"]
    assert chunks[0].text.startswith("a0 a1")
    assert chunks[1].heading_trail == ["Title", "Section B"]
    assert chunks[1].text.startswith("b0 b1")


def test_nested_headings_pop_back_to_shallower_sibling() -> None:
    text = (
        "# Title\n\n"
        "## Parent\n\n"
        "### Child\n\n"
        f"{_numbered_words('c', 3)}\n\n"
        "## Sibling\n\n"
        f"{_numbered_words('s', 3)}\n"
    )
    chunks = chunk_document("doc.md", text, chunk_words=100, overlap_words=10)

    assert chunks[0].heading_trail == ["Title", "Parent", "Child"]
    # Sibling is back at the "## " level, so "Child" must not leak into its trail.
    assert chunks[1].heading_trail == ["Title", "Sibling"]


def test_word_budget_chunks_overlap_at_the_boundary() -> None:
    # 25 words, budget of 10 with overlap of 3 -> step of 7 -> windows at
    # [0:10], [7:17], [14:24], [21:25].
    text = "# Section\n\n" + _numbered_words("w", 25)
    chunks = chunk_document("doc.md", text, chunk_words=10, overlap_words=3)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len(chunks) == 4

    words_0 = chunks[0].text.split()
    words_1 = chunks[1].text.split()
    assert len(words_0) == 10
    # Last 3 words of chunk 0 must equal the first 3 words of chunk 1.
    assert words_0[-3:] == words_1[:3]

    # Every word from the source text shows up somewhere in the chunks.
    all_words = {w for c in chunks for w in c.text.split()}
    assert all_words == {f"w{i}" for i in range(25)}


def test_short_section_is_a_single_chunk_with_no_split() -> None:
    text = "# Section\n\n" + _numbered_words("w", 5)
    chunks = chunk_document("doc.md", text, chunk_words=180, overlap_words=40)
    assert len(chunks) == 1
    assert chunks[0].text == _numbered_words("w", 5)


def test_headingless_text_falls_back_to_one_section() -> None:
    text = _numbered_words("p", 5)
    chunks = chunk_document("notes.txt", text, chunk_words=100, overlap_words=10)
    assert len(chunks) == 1
    assert chunks[0].heading_trail == []
    assert chunks[0].source == "notes.txt"


def test_chunk_ids_are_stable_across_repeated_chunking() -> None:
    text = "# Section\n\n" + _numbered_words("w", 25)
    first = chunk_document("doc.md", text, chunk_words=10, overlap_words=3)
    second = chunk_document("doc.md", text, chunk_words=10, overlap_words=3)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    # Ids are also unique within a document.
    assert len({c.chunk_id for c in first}) == len(first)


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_document("doc.md", "# X\n\nhello world", chunk_words=10, overlap_words=10)
