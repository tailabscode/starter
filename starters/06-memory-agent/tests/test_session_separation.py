"""Session memory is genuinely ephemeral: it never carries facts across sessions.

Only the persistent store does. This test drives the shared `run_turn` core
loop directly (in-process, fast) to prove the separation at the data-structure
level; `test_demo_end_to_end.py` proves the same thing across real processes.
"""

from memory_agent.chat import run_turn
from memory_agent.embedder import HashingEmbedder
from memory_agent.llm import StubClient
from memory_agent.session import SessionMemory
from memory_agent.store import MemoryStore


def test_new_session_starts_with_empty_messages() -> None:
    session = SessionMemory(session_id="brand-new")
    assert session.messages == []


def test_session_history_alone_does_not_contain_facts_from_a_prior_session(tmp_path) -> None:
    llm_client = StubClient()
    with MemoryStore(str(tmp_path / "memories.db"), HashingEmbedder(dim=64)) as store:
        session_1 = SessionMemory(session_id="session-1")
        run_turn(
            session_1,
            store,
            llm_client,
            "I prefer Python over JavaScript for backend work.",
            top_k=3,
            dedup_threshold=0.82,
            max_memories_per_turn=3,
        )

        # A brand new session object: no history loaded from anywhere.
        session_2 = SessionMemory(session_id="session-2")
        assert session_2.messages == []  # empty before the turn even starts

        run_turn(
            session_2,
            store,
            llm_client,
            "Do I prefer Python and where do I live?",
            top_k=3,
            dedup_threshold=0.82,
            max_memories_per_turn=3,
        )

        # Session 2's own conversation history is just its own one exchange --
        # "Python" never appears there as something *session 2* said or was told
        # directly; it only reaches session 2 through the retrieved-memory block
        # baked into the system prompt, not through session_2.messages content
        # inherited from session 1.
        session_2_user_messages = [m["content"] for m in session_2.messages if m["role"] == "user"]
        assert session_2_user_messages == ["Do I prefer Python and where do I live?"]

        # And yet the reply is correct, because persistent memory bridged it.
        assert "Python" in session_2.messages[-1]["content"]

        # The persistent store, not session memory, is what carried the fact.
        persisted_contents = " ".join(r.content for r in store.list_all())
        assert "Python" in persisted_contents
