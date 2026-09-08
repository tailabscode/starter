import pytest

from agentic_rag.errors import CorpusNotFoundError, UnknownChunkError
from agentic_rag.retrieval import HybridRetriever


def test_search_finds_aurora_team_for_caching_question(retriever: HybridRetriever) -> None:
    results = retriever.search("who rebuilt the caching layer", top_k=3)
    assert results
    assert any("overview" in r["source"] for r in results)
    for hit in results:
        assert hit["chunk_id"] in retriever.valid_chunk_ids()


def test_search_second_hop_finds_teams_doc(retriever: HybridRetriever) -> None:
    results = retriever.search("Aurora team", top_k=3)
    assert any(r["source"] == "teams.md" for r in results)


def test_list_topics_covers_every_document(retriever: HybridRetriever) -> None:
    topics = retriever.list_topics()
    doc_ids = {t["doc_id"] for t in topics}
    assert doc_ids == {"overview", "architecture", "deployment", "teams", "incidents", "roadmap"}


def test_fetch_chunk_returns_neighbours(retriever: HybridRetriever) -> None:
    any_id = next(iter(retriever.valid_chunk_ids()))
    result = retriever.fetch_chunk(any_id)
    assert result["chunk_id"] == any_id
    assert result["context"]
    assert any(c["chunk_id"] == any_id for c in result["context"])


def test_fetch_unknown_chunk_raises(retriever: HybridRetriever) -> None:
    with pytest.raises(UnknownChunkError):
        retriever.fetch_chunk("does-not-exist::99")


def test_save_and_load_round_trip(retriever: HybridRetriever, tmp_path) -> None:
    path = tmp_path / "index.json"
    retriever.save(path)
    reloaded = HybridRetriever.load(path)
    assert reloaded.valid_chunk_ids() == retriever.valid_chunk_ids()
    assert reloaded.search("Aurora team", top_k=1)[0]["chunk_id"]


def test_empty_data_dir_raises(tmp_path) -> None:
    with pytest.raises(CorpusNotFoundError):
        HybridRetriever.from_data_dir(tmp_path)
