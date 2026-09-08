"""Retrieval: relevant memories rank above irrelevant ones for a given query."""

from memory_agent.embedder import HashingEmbedder
from memory_agent.store import MemoryStore


def test_retrieval_ranks_relevant_memory_above_irrelevant_ones(tmp_path) -> None:
    # Phrasing here mirrors what the offline stub extractor actually produces
    # (it swaps the leading pronoun but doesn't reconjugate verbs) -- the
    # hashing embedder has no stemming, so "prefer" and "prefers" are
    # different tokens. This test is about vocabulary overlap, not grammar.
    with MemoryStore(str(tmp_path / "memories.db"), HashingEmbedder(dim=64)) as store:
        store.create("User prefer Python over JavaScript for backend work.", "preference", "s1")
        store.create("User live in Berlin.", "fact", "s1")
        store.create("User is allergic to peanuts.", "fact", "s1")
        store.create("User's favorite color is teal.", "preference", "s1")

        results = store.retrieve_top_k("Do I prefer Python and where do I live?", k=2)

        assert len(results) == 2
        contents = [record.content for record, _score in results]
        assert "User prefer Python over JavaScript for backend work." in contents
        assert "User live in Berlin." in contents
        # both irrelevant memories are excluded from the top 2
        assert "User is allergic to peanuts." not in contents
        assert "User's favorite color is teal." not in contents

        # scores are actually sorted descending, not just filtered
        scores = [score for _record, score in results]
        assert scores == sorted(scores, reverse=True)


def test_retrieval_returns_nothing_when_no_memory_shares_vocabulary(tmp_path) -> None:
    with MemoryStore(str(tmp_path / "memories.db"), HashingEmbedder(dim=64)) as store:
        store.create("User is allergic to peanuts.", "fact", "s1")
        results = store.retrieve_top_k("What is the capital of France?", k=3)
        assert results == []


def test_retrieval_touches_last_accessed_at_on_returned_memories(tmp_path) -> None:
    with MemoryStore(str(tmp_path / "memories.db"), HashingEmbedder(dim=64)) as store:
        record = store.create("User prefers tea over coffee.", "preference", "s1")
        original_access_time = record.last_accessed_at

        store.retrieve_top_k("What does the user prefer, tea or coffee?", k=1)

        refreshed = store.get(record.id)
        assert refreshed is not None
        assert refreshed.last_accessed_at >= original_access_time
