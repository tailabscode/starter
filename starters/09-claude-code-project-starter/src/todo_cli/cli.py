"""Command-line interface for todo_cli.

Subcommands: add, list, complete, remove. Storage location resolves in
order: --db flag > TODO_DB_PATH env var > ~/.todo_cli/todo.db.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from todo_cli.storage import Task, TaskStore

DEFAULT_DB_PATH = Path.home() / ".todo_cli" / "todo.db"


def _resolve_db_path(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    env_value = os.environ.get("TODO_DB_PATH")
    if env_value:
        return Path(env_value)
    return DEFAULT_DB_PATH


def _format_task(task: Task) -> str:
    mark = "x" if task.done else " "
    return f"[{mark}] {task.id}: {task.description}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="todo", description="A tiny SQLite-backed todo list.")
    parser.add_argument(
        "--db",
        help="Path to the SQLite database file (default: ~/.todo_cli/todo.db or $TODO_DB_PATH)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="Add a new task")
    add_p.add_argument("description", help="Task description")

    list_p = sub.add_parser("list", help="List tasks")
    list_p.add_argument("--pending", action="store_true", help="Only show tasks that are not done")

    complete_p = sub.add_parser("complete", help="Mark a task as done")
    complete_p.add_argument("task_id", type=int)

    remove_p = sub.add_parser("remove", help="Delete a task")
    remove_p.add_argument("task_id", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = TaskStore(_resolve_db_path(args.db))

    try:
        if args.command == "add":
            task = store.add(args.description)
            print(f"Added task {task.id}: {task.description}")
        elif args.command == "list":
            tasks = store.list(include_done=not args.pending)
            if not tasks:
                print('No tasks yet. Add one with: todo add "..."')
            for task in tasks:
                print(_format_task(task))
        elif args.command == "complete":
            task = store.complete(args.task_id)
            print(f"Completed task {task.id}: {task.description}")
        elif args.command == "remove":
            store.remove(args.task_id)
            print(f"Removed task {args.task_id}")
    except (ValueError, KeyError) as exc:
        # KeyError.__str__ wraps the message in repr() quotes; args[0] is the
        # plain message we raised it with, for both ValueError and KeyError.
        print(f"Error: {exc.args[0]}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
