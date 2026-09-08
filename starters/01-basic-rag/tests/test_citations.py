"""Citation resolution: real markers resolve, unresolvable ones are reported
(never silently dropped)."""

from __future__ import annotations

from basic_rag.chunking import Chunk
from basic_rag.generation import build_context_blocks, make_answer_result, resolve_citations


def _chunk(chunk_id: str, source: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source=source, heading_trail=["H"], chunk_index=0, text="t")


def test_valid_markers_resolve_to_real_chunks() -> None:
    blocks = build_context_blocks([_chunk("id-a", "a.md"), _chunk("id-b", "b.md")])
    citations = resolve_citations("Fact one [1]. Fact two [2].", blocks)

    assert [c.marker for c in citations] == [1, 2]
    assert all(c.resolved for c in citations)
    assert citations[0].chunk_id == "id-a"
    assert citations[0].source == "a.md"
    assert citations[1].chunk_id == "id-b"


def test_unresolvable_marker_is_reported_not_dropped() -> None:
    blocks = build_context_blocks([_chunk("id-a", "a.md"), _chunk("id-b", "b.md")])
    citations = resolve_citations("Fact one [1]. A hallucinated fact [5].", blocks)

    assert [c.marker for c in citations] == [1, 5]
    marker_5 = citations[1]
    assert marker_5.resolved is False
    assert marker_5.chunk_id is None
    assert marker_5.source is None


def test_repeated_marker_only_counted_once() -> None:
    blocks = build_context_blocks([_chunk("id-a", "a.md")])
    citations = resolve_citations("[1] and again [1] and once more [1].", blocks)
    assert len(citations) == 1
    assert citations[0].marker == 1


def test_no_markers_in_answer_yields_no_citations() -> None:
    blocks = build_context_blocks([_chunk("id-a", "a.md")])
    citations = resolve_citations("I don't know based on the given context.", blocks)
    assert citations == []


def test_make_answer_result_collects_unresolved_markers() -> None:
    blocks = build_context_blocks([_chunk("id-a", "a.md")])
    result = make_answer_result("See [1] and also [9].", blocks)
    assert result.unresolved_markers == [9]
    assert result.text == "See [1] and also [9]."
