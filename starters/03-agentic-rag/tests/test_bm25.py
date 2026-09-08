from agentic_rag.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Aurora's Cache, Rebuilt!") == ["aurora", "s", "cache", "rebuilt"]


def test_bm25_scores_relevant_document_highest() -> None:
    docs = [
        "the aurora team owns the shared cache",
        "deployments use a canary rollout process",
        "the alert publisher deduplicates alerts",
    ]
    index = BM25Index(docs)
    scores = index.score("aurora cache")
    assert scores[0] == max(scores)
    assert scores[0] > 0


def test_bm25_zero_for_empty_query_or_corpus() -> None:
    assert BM25Index(["a document"]).score("") == [0.0]
    assert BM25Index([]).score("anything") == []


def test_bm25_term_absent_scores_zero_contribution() -> None:
    index = BM25Index(["only mentions widgets"])
    assert index.score("gadgets")[0] == 0.0
