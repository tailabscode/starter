import json

from agentic_rag.cli import main


def test_query_offline_end_to_end(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AGENTIC_RAG_TRACE_PATH", str(tmp_path / "trace.json"))
    exit_code = main(["query", "Aurora team", "--offline"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "offline stub" in out

    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["query"] == "Aurora team"
    assert trace["steps"]


def test_plan_offline_prints_classification(capsys) -> None:
    exit_code = main(["plan", "What is 2 + 2?", "--offline"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["classification"] == "NO_RETRIEVAL"


def test_trace_reports_missing_run(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AGENTIC_RAG_TRACE_PATH", str(tmp_path / "does-not-exist.json"))
    exit_code = main(["trace"])
    assert exit_code == 1
    assert "No trace found" in capsys.readouterr().out
