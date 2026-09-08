"""Reciprocal Rank Fusion ordering, and hybrid retrieval returning a mix
of text and image items."""

from __future__ import annotations

from multimodal_rag.embeddings import HashingEmbedder
from multimodal_rag.index import Index, IndexMetadata
from multimodal_rag.items import Item
from multimodal_rag.retrieval import HybridRetriever, reciprocal_rank_fusion


def test_fusion_prefers_ids_ranked_well_by_both_lists() -> None:
    bm25_ranking = ["a", "b", "c"]
    dense_ranking = ["c", "a", "b"]
    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)
    assert [doc_id for doc_id, _ in fused] == ["a", "c", "b"]


def test_fusion_scores_match_hand_computed_values() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=10))
    expected = 1 / 11 + 1 / 12
    assert fused["a"] == expected
    assert fused["b"] == expected


def _mixed_index() -> Index:
    items = [
        Item(
            item_id="text-1",
            kind="text",
            source="notes.md",
            text="the quarterly revenue chart is drawn mostly in blue bars",
            heading_trail=["Notes"],
        ),
        Item(
            item_id="text-2",
            kind="text",
            source="other.md",
            text="unrelated onboarding steps for new warehouse staff",
            heading_trail=["Onboarding"],
        ),
        Item(
            item_id="image-1",
            kind="image",
            source="chart.png",
            text="100x70px image. dominant color overall rgb(255, 255, 255). "
            "dominant non-background color rgb(30, 90, 200) blue",
            heading_trail=[],
            image_path="/tmp/chart.png",
            media_type="image/png",
        ),
    ]
    embedder = HashingEmbedder()
    vectors = embedder.embed([item.text for item in items])
    metadata = IndexMetadata(embedder_name=embedder.name, embedder_dim=embedder.dim)
    return Index(items=items, vectors=vectors, metadata=metadata)


def test_hybrid_retrieval_returns_a_mix_of_text_and_image_items() -> None:
    index = _mixed_index()
    retriever = HybridRetriever(index=index, embedder=HashingEmbedder())

    # Only 2 of 3 slots: this only passes if retrieval actually ranks the
    # relevant text chunk AND the relevant image above the unrelated text.
    retrieved = retriever.retrieve("blue revenue chart", top_k=2, rrf_k=60)
    ranked_ids = [r.item_id for r in retrieved]
    retrieved_kinds = {index.get_item(item_id).kind for item_id in ranked_ids}

    assert retrieved_kinds == {"text", "image"}
    assert "text-2" not in ranked_ids  # the unrelated onboarding chunk loses out
    assert "text-1" in ranked_ids
    assert "image-1" in ranked_ids
