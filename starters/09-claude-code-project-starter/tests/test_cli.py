"""Tests for the argparse CLI, driven through main() rather than subprocess
so failures show a normal Python traceback instead of an opaque exit code.
"""

from pathlib import Path

import pytest

from todo_cli.cli import main


def test_add_and_list_via_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "todo.db")
    assert main(["--db", db, "add", "Buy milk"]) == 0
    assert main(["--db", db, "list"]) == 0
    out = capsys.readouterr().out
    assert "Buy milk" in out
    assert "[ ] 1: Buy milk" in out


def test_complete_via_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "todo.db")
    main(["--db", db, "add", "Write tests"])
    capsys.readouterr()
    assert main(["--db", db, "complete", "1"]) == 0
    out = capsys.readouterr().out
    assert "Completed task 1" in out


def test_complete_unknown_id_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = str(tmp_path / "todo.db")
    exit_code = main(["--db", db, "complete", "5"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "No task with id 5" in err


def test_list_empty_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "todo.db")
    main(["--db", db, "list"])
    out = capsys.readouterr().out
    assert "No tasks yet" in out


def test_remove_via_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "todo.db")
    main(["--db", db, "add", "Temp"])
    capsys.readouterr()
    assert main(["--db", db, "remove", "1"]) == 0
    main(["--db", db, "list"])
    out = capsys.readouterr().out
    assert "No tasks yet" in out


def test_list_pending_hides_done(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "todo.db")
    main(["--db", db, "add", "Task A"])
    main(["--db", db, "add", "Task B"])
    main(["--db", db, "complete", "1"])
    capsys.readouterr()
    main(["--db", db, "list", "--pending"])
    out = capsys.readouterr().out
    assert "Task A" not in out
    assert "Task B" in out
