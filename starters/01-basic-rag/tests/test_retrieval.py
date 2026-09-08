"""Tests for Reciprocal Rank Fusion ordering."""

from __future__ import annotations

from basic_rag.retrieval import reciprocal_rank_fusion


def test_fusion_prefers_ids_ranked_well_by_both_lists() -> None:
    # "a" is 1st then 2nd (avg rank ~1.5), "c" is 3rd then 1st (avg rank ~2),
    # "b" is 2nd then 3rd (avg rank ~2.5). RRF should reflect that ordering.
    bm25_ranking = ["a", "b", "c"]
    dense_ranking = ["c", "a", "b"]

    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)
    fused_ids = [doc_id for doc_id, _ in fused]

    assert fused_ids == ["a", "c", "b"]


def test_fusion_scores_match_hand_computed_values() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=10))
    # a: rank 1 in list 1 (1/11) + rank 2 in list 2 (1/12)
    # b: rank 2 in list 1 (1/12) + rank 1 in list 2 (1/11)
    expected = 1 / 11 + 1 / 12
    assert fused["a"] == expected
    assert fused["b"] == expected  # symmetric case: identical scores


def test_id_present_in_only_one_ranking_still_scores() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["a"]], k=60))
    assert fused["a"] == 1 / 61 + 1 / 61
    assert fused["b"] == 1 / 62
    assert fused["a"] > fused["b"]


def test_higher_k_flattens_the_score_gap_between_ranks() -> None:
    small_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    large_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    gap_small_k = small_k["a"] - small_k["b"]
    gap_large_k = large_k["a"] - large_k["b"]
    assert gap_small_k > gap_large_k > 0
