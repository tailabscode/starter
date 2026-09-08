"""CLI surface: memories, forget, and the max-memories-per-turn bound."""

from memory_agent import cli


def test_cli_chat_then_memories_then_forget(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_AGENT_DB", str(tmp_path / "cli.db"))

    cli.main(["chat", "I prefer Python over JavaScript for backend work.", "--offline"])
    capsys.readouterr()

    cli.main(["memories"])
    memories_out = capsys.readouterr().out
    assert "preference" in memories_out
    assert "Python" in memories_out

    memory_id = memories_out.splitlines()[0].split()[0]

    exit_code = cli.main(["forget", memory_id])
    forget_out = capsys.readouterr().out
    assert exit_code == 0
    assert f"forgot memory {memory_id}" in forget_out

    cli.main(["memories"])
    after_out = capsys.readouterr().out
    assert "no memories stored yet" in after_out


def test_cli_forget_unknown_id_returns_error(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_AGENT_DB", str(tmp_path / "cli.db"))
    exit_code = cli.main(["forget", "does-not-exist"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "no memory with id" in err


def test_extraction_proposals_are_capped_per_turn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_AGENT_DB", str(tmp_path / "cli.db"))
    monkeypatch.setenv("MAX_MEMORIES_PER_TURN", "1")

    # Two sentences would normally propose two memories; the cap limits it to one.
    cli.main(["chat", "I live in Berlin. I'm allergic to peanuts.", "--offline"])

    from memory_agent.config import load_config
    from memory_agent.embedder import HashingEmbedder
    from memory_agent.store import MemoryStore

    config = load_config()
    with MemoryStore(config.db_path, HashingEmbedder(dim=config.embedding_dim)) as store:
        assert len(store.list_all()) == 1
