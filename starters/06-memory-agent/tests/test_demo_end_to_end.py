"""The full offline two-session demo, run for real across two subprocesses.

This is the entire point of the starter: session 1 (its own process) writes
facts to disk and exits; session 2 (a different process, empty conversation
history) must answer correctly using only what is on disk.
"""

from memory_agent.demo import run_two_session_demo


def test_two_session_demo_produces_the_correct_answer_offline(tmp_path) -> None:
    db_path = str(tmp_path / "demo.db")

    result = run_two_session_demo(db_path=db_path, offline=True)

    assert len(result.turns) == 3
    assert result.turns[0].session_id == "session-1"
    assert result.turns[1].session_id == "session-1"
    assert result.turns[2].session_id == "session-2"

    session2_stdout = result.session2_reply.stdout
    assert "session_id=session-2" in session2_stdout

    reply_line = next(
        line for line in session2_stdout.splitlines() if line.startswith("assistant: ")
    )
    assert "Python" in reply_line
    assert "Berlin" in reply_line

    # Session 2's own transcript never shows a "you:" line establishing these
    # facts -- the only "you:" line is the question itself.
    you_lines = [line for line in session2_stdout.splitlines() if line.startswith("you: ")]
    assert you_lines == ["you: Do I prefer Python and where do I live?"]
