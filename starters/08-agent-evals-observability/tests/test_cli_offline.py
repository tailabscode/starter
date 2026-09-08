"""End-to-end CLI tests, forced offline via --offline so they never touch the network.
The runs/ output directory is monkeypatched to tmp_path so tests never write into
the repo's real runs/ folder."""

from agent_evals import cli as cli_module


def test_run_offline_passes_and_writes_trace(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "DEFAULT_RUNS_DIR", tmp_path)
    exit_code = cli_module.main(["--offline", "run", "--run-id", "cli-test-a"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "pass_rate=1.00" in out
    assert "-> PASS" in out
    assert (tmp_path / "cli-test-a.jsonl").exists()


def test_run_offline_fails_when_threshold_unreachable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "DEFAULT_RUNS_DIR", tmp_path)
    exit_code = cli_module.main(
        ["--offline", "run", "--run-id", "cli-test-b", "--threshold", "1.5"]
    )
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "-> FAIL" in out


def test_show_prints_case_details(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "DEFAULT_RUNS_DIR", tmp_path)
    cli_module.main(["--offline", "run", "--run-id", "cli-test-c"])
    capsys.readouterr()

    exit_code = cli_module.main(["show", "cli-test-c"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "kb-refund" in out
    assert "calc-div-zero" in out


def test_compare_reports_no_changes_between_identical_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "DEFAULT_RUNS_DIR", tmp_path)
    cli_module.main(["--offline", "run", "--run-id", "run-x"])
    capsys.readouterr()
    cli_module.main(["--offline", "run", "--run-id", "run-y"])
    capsys.readouterr()

    exit_code = cli_module.main(["compare", "run-x", "run-y"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "0 regression(s)" in out


def test_compare_missing_run_errors_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "DEFAULT_RUNS_DIR", tmp_path)
    exit_code = cli_module.main(["compare", "nope-a", "nope-b"])
    err = capsys.readouterr().err

    assert exit_code == 1
    assert "error:" in err
