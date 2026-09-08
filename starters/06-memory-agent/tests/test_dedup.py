"""De-duplication: a near-identical memory updates in place instead of duplicating."""

from memory_agent.embedder import HashingEmbedder
from memory_agent.store import MemoryStore

THRESHOLD = 0.82


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "memories.db"), HashingEmbedder(dim=64))


def test_near_identical_content_updates_in_place(tmp_path) -> None:
    with _store(tmp_path) as store:
        first, action1 = store.upsert_with_dedup(
            "User prefers Python for backend development.", "preference", "s1", THRESHOLD
        )
        assert action1 == "create"

        second, action2 = store.upsert_with_dedup(
            "User prefers Python for backend development work.", "preference", "s1", THRESHOLD
        )
        assert action2 == "update"
        assert second.id == first.id  # same record, not a new one

        all_records = store.list_all()
        assert len(all_records) == 1
        assert all_records[0].content == "User prefers Python for backend development work."


def test_clearly_different_content_creates_a_new_record(tmp_path) -> None:
    with _store(tmp_path) as store:
        first, _ = store.upsert_with_dedup(
            "User prefers Python for backend development.", "preference", "s1", THRESHOLD
        )
        second, action = store.upsert_with_dedup(
            "User dislikes cilantro in food.", "preference", "s1", THRESHOLD
        )
        assert action == "create"
        assert second.id != first.id
        assert len(store.list_all()) == 2


def test_dedup_threshold_is_respected_at_the_boundary(tmp_path) -> None:
    with _store(tmp_path) as store:
        store.upsert_with_dedup("User lives in Berlin.", "fact", "s1", THRESHOLD)
        # A high threshold means only a near-exact rephrasing counts as a duplicate.
        _, action = store.upsert_with_dedup(
            "User once visited Berlin on a trip.", "episodic", "s1", threshold=0.99
        )
        assert action == "create"
