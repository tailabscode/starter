"""Ranking sanity checks for the from-scratch Okapi BM25 implementation."""

from __future__ import annotations

from basic_rag.bm25 import BM25, tokenize

DOCS = [
    "the wellness stipend covers gym memberships and therapy",
    "parental leave is sixteen weeks for the primary caregiver",
    "the home office equipment stipend is a one time payment",
    "public holidays vary between the Porto and Nairobi offices",
]


def test_query_ranks_the_matching_document_first() -> None:
    bm25 = BM25([tokenize(d) for d in DOCS])
    ranking = bm25.rank(tokenize("wellness stipend"))
    assert ranking[0] == 0  # doc 0 is the only one mentioning "wellness"


def test_repeated_term_frequency_increases_score() -> None:
    docs = [
        "stipend stipend stipend stipend",
        "stipend appears once here",
    ]
    bm25 = BM25([tokenize(d) for d in docs])
    scores = bm25.score_all(tokenize("stipend"))
    assert scores[0] > scores[1] > 0


def test_term_absent_from_corpus_contributes_nothing() -> None:
    bm25 = BM25([tokenize(d) for d in DOCS])
    scores = bm25.score_all(tokenize("zzznotinanydoc"))
    assert scores == [0.0, 0.0, 0.0, 0.0]


def test_common_term_has_lower_idf_than_rare_term() -> None:
    # "the" appears in 3/4 docs, "gym" appears in 1/4 -> "gym" should carry
    # more discriminative weight (higher idf).
    bm25 = BM25([tokenize(d) for d in DOCS])
    assert bm25.idf["gym"] > bm25.idf["the"]


def test_empty_corpus_does_not_crash() -> None:
    bm25 = BM25([])
    assert bm25.rank(tokenize("anything")) == []
