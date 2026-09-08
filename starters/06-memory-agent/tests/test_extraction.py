"""Memory extraction: the offline stub proposes sensible actions on a scripted exchange."""

from memory_agent.llm import StubClient


def test_extraction_proposes_a_preference_and_a_fact() -> None:
    exchange = [
        {"role": "user", "content": "I prefer Python over JavaScript for backend work."},
        {"role": "assistant", "content": "[stub] Noted."},
    ]
    proposals = StubClient().extract_memories(exchange)

    assert len(proposals) == 1
    assert proposals[0]["action"] == "create"
    assert proposals[0]["category"] == "preference"
    assert "Python" in proposals[0]["content"]


def test_extraction_splits_multiple_sentences_into_multiple_proposals() -> None:
    exchange = [
        {"role": "user", "content": "I live in Berlin. I'm allergic to peanuts."},
        {"role": "assistant", "content": "[stub] Got it."},
    ]
    proposals = StubClient().extract_memories(exchange)

    assert len(proposals) == 2
    assert all(p["category"] == "fact" for p in proposals)
    contents = " ".join(p["content"] for p in proposals)
    assert "Berlin" in contents
    assert "peanuts" in contents


def test_extraction_ignores_small_talk_with_nothing_memorable() -> None:
    exchange = [
        {"role": "user", "content": "What's the weather like today?"},
        {"role": "assistant", "content": "[stub] I don't have live weather data."},
    ]
    proposals = StubClient().extract_memories(exchange)
    assert proposals == []


def test_extraction_does_not_fire_on_a_question_that_mentions_the_marker_mid_sentence() -> None:
    # "Do I prefer..." is a question, not a first-person statement -- the
    # marker only counts when it anchors the start of the sentence.
    exchange = [
        {"role": "user", "content": "Do I prefer Python and where do I live?"},
        {"role": "assistant", "content": "[stub] Based on what I remember..."},
    ]
    proposals = StubClient().extract_memories(exchange)
    assert proposals == []


def test_extraction_only_reads_user_messages_not_the_assistant_reply() -> None:
    exchange = [
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I prefer to stay neutral, but thanks for asking!"},
    ]
    proposals = StubClient().extract_memories(exchange)
    assert proposals == []
