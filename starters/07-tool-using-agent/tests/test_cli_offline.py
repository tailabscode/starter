"""End-to-end CLI tests, forced offline via --offline so they never touch the network."""

from tool_using_agent.cli import main


def test_list_tools_lists_all_four_tools(capsys):
    exit_code = main(["list-tools"])
    out = capsys.readouterr().out
    assert exit_code == 0
    for name in ("calculator", "unit_convert", "lookup_fact", "http_style_adapter"):
        assert name in out


def test_chat_offline_calculator(capsys):
    exit_code = main(["--offline", "chat", "What is 6 * 7?"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "42" in out


def test_chat_offline_weather(capsys):
    exit_code = main(["--offline", "chat", "What's the weather in Paris?"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Sunny" in out or "Paris" in out


def test_chat_offline_never_touches_network_even_without_a_key(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")  # would normally require a key
    exit_code = main(["--offline", "chat", "Tell me about dna"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "DNA" in out or "genetic" in out.lower()


def test_max_steps_bound_from_cli(capsys):
    exit_code = main(["--offline", "--max-steps", "1", "chat", "What is 6 * 7?"])
    out = capsys.readouterr().out
    # With max_steps=1 the stub's first turn (a tool call) already consumes the
    # single allowed step, so the loop must stop with the bound message.
    assert exit_code == 1
    assert "max_steps" in out
