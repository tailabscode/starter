"""Tests for the TaskStore SQLite layer. All tests use tmp_path -- no shared state."""

from pathlib import Path

import pytest

from todo_cli.storage import TaskStore


def test_add_and_list(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "todo.db")
    task = store.add("Buy milk")
    assert task.id == 1
    assert task.description == "Buy milk"
    assert task.done is False

    tasks = store.list()
    assert len(tasks) == 1
    assert tasks[0].description == "Buy milk"


def test_add_strips_whitespace(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "todo.db")
    task = store.add("  Buy milk  ")
    assert task.description == "Buy milk"


def test_add_rejects_empty_description(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "todo.db")
    with pytest.raises(ValueError):
        store.add("   ")


def test_complete_marks_done(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "todo.db")
    task = store.add("Write tests")
    completed = store.complete(task.id)
    assert completed.done is True
    assert store.list(include_done=False) == []


def test_complete_unknown_id_raises(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "todo.db")
    with pytest.raises(KeyError):
        store.complete(999)


def test_remove_deletes_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "todo.db")
    task = store.add("Temporary")
    store.remove(task.id)
    assert store.list() == []


def test_remove_unknown_id_raises(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "todo.db")
    with pytest.raises(KeyError):
        store.remove(42)


def test_persistence_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "todo.db"
    TaskStore(db_path).add("Persisted task")
    reopened = TaskStore(db_path)
    assert len(reopened.list()) == 1


def test_list_order_is_by_id(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "todo.db")
    store.add("first")
    store.add("second")
    store.add("third")
    ids = [t.id for t in store.list()]
    assert ids == sorted(ids)
