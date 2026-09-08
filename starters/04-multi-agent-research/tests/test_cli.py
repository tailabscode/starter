"""Offline end-to-end: the CLI runs the full pipeline with no network, then lists/shows it."""

from __future__ import annotations

import json

from multi_agent_research.cli import main


def test_research_offline_end_to_end(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MAR_RUNS_DIR", str(tmp_path))

    exit_code = main(["research", "How risky is the EV battery supply chain?", "--offline"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Run id:" in out
    assert "offline stub" in out
    assert "Sources used:" in out


def test_list_and_show_after_a_research_run(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MAR_RUNS_DIR", str(tmp_path))
    main(["research", "What reduces cobalt exposure?", "--offline"])
    capsys.readouterr()  # discard the research output

    assert main(["list"]) == 0
    listing = capsys.readouterr().out
    assert listing.strip()
    run_id = listing.split()[0]

    assert main(["show", run_id]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == run_id
    assert payload["final_report"]
    assert payload["termination_reason"] in {"sufficient", "max_rounds_reached", "no_gaps_named"}


def test_list_with_no_runs_says_so(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MAR_RUNS_DIR", str(tmp_path))

    assert main(["list"]) == 0
    assert "No runs found" in capsys.readouterr().out


def test_show_missing_run_reports_error(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MAR_RUNS_DIR", str(tmp_path))

    exit_code = main(["show", "does-not-exist"])

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err
