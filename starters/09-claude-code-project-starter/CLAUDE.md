# CLAUDE.md

Instructions for Claude Code (or any coding agent) working in this repository.
This file is deliberately concrete — every command below was run against the
actual code in this repo and works as written. Copy the *structure*, not the
literal content, into your own project.

## Project overview

`todo_cli` is a small, real CLI todo-list manager: add, list, complete, and
remove tasks, backed by SQLite. It exists in this starter as a realistic demo
target — a project small enough to read end to end in five minutes, real
enough that delegating a task to Claude Code against it is representative of
real work. It has no network calls and needs no API key.

## Architecture

Three modules, each with one job: `cli.py` parses arguments and prints
output, `storage.py` owns the SQLite schema and all reads/writes, and
`__main__.py` exists only so `python -m todo_cli` works. `cli.py` never
touches `sqlite3` directly, and `storage.py` never calls `print()` — that
split is what keeps the CLI layer testable through `main()` without spinning
up a subprocess, and the storage layer testable without argparse in the way.

```
starters/09-claude-code-project-starter/
├── CLAUDE.md                    # this file
├── README.md                    # starter overview + how to read this example
├── pyproject.toml               # hatchling build, [project.scripts] todo=...
├── .env.example                 # documents the one optional env var
├── .claude/
│   └── settings.json            # example scoped tool permissions (see README)
├── docs/
│   └── development-workflow.md  # worked example: delegating a real change
├── src/todo_cli/
│   ├── __init__.py
│   ├── __main__.py              # `python -m todo_cli`
│   ├── cli.py                   # argparse, subcommands, output formatting
│   └── storage.py               # TaskStore: SQLite schema + CRUD
└── tests/
    ├── test_storage.py          # TaskStore, via tmp_path databases
    └── test_cli.py              # main(argv), via capsys
```

## Setup, build, test, lint

Run from `starters/09-claude-code-project-starter/` (this directory):

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[dev]" --python .venv

uv run --python .venv ruff check .            # lint
uv run --python .venv ruff format --check .   # format check (uv run ... ruff format . to fix)
uv run --python .venv pytest -q               # 15 tests, offline, <1s

uv run --python .venv todo add "Buy milk"     # exercise the CLI for real
uv run --python .venv todo list
```

If you install into your normal environment instead of a throwaway `.venv`
(e.g. `pip install -e ".[dev]"` at the system/user level), drop the
`--python .venv` and `uv run --python .venv` prefixes and just use `todo`,
`pytest`, `ruff` directly.

## Coding conventions used in this repo

- Python 3.11+, full type hints on public functions, no bare `except:`.
- `storage.py` raises `ValueError` for bad input (empty description) and
  `KeyError` for "no such task id" — `cli.py` is the only place that catches
  them and turns them into a printed message plus exit code 1.
- Every `TaskStore` method opens and closes its own connection (see the
  docstring in `storage.py` for why); don't "optimize" this into a single
  long-lived connection without a reason tied to a measured problem.
- Ruff config lives in `pyproject.toml`: `line-length = 100`,
  `select = ["E", "F", "I", "UP", "B"]`. Run `ruff format` before committing;
  don't hand-format around it.
- Tests use `tmp_path` for storage tests and `capsys` for CLI tests — never a
  shared database file, never `subprocess` (it would hide real tracebacks).

## Dos and don'ts for the agent

**Do:**
- Read `src/todo_cli/storage.py` and `src/todo_cli/cli.py` in full before
  changing either — they're under 120 lines each, there's no excuse to guess.
- Add a test alongside any behavior change, in the matching test file.
- Run the full self-check loop above (lint, format check, tests) before
  reporting a change as done.
- Keep new CLI flags consistent with the existing ones: long-form
  (`--pending`, not `-p`), `argparse` `choices=` for closed sets of values.

**Don't:**
- Don't add a dependency for something the standard library already does
  (this project's only runtime import outside the stdlib is... none).
- Don't change the SQLite schema without a migration story, even an
  informal one written in the PR description — this is a toy project but
  the habit is the point.
- Don't reach for `subprocess` in tests when calling `main(argv)` directly
  gives you a real traceback on failure.
- Don't touch `.claude/settings.json` permissions without calling it out
  explicitly in your summary — permission scope changes are security-relevant.

## Safe delegation

What Claude Code can do **autonomously** in this repo, without asking first:
- Add/modify tests in `tests/`.
- Add a new CLI flag or subcommand that doesn't change existing behavior for
  users who don't pass it (e.g. an additive `--priority` flag with a
  backward-compatible default — see `docs/development-workflow.md` for a
  full worked example of exactly this).
- Fix a lint/format failure.
- Update docstrings/comments to match code that already changed.

What needs **explicit confirmation** before Claude Code acts:
- Changing the SQLite schema in a way that isn't purely additive (renaming
  or dropping a column, changing a type) — this can silently drop user data
  in `~/.todo_cli/todo.db`.
- Changing the default database path or the CLI's default behavior for
  existing flags (e.g. making `list` hide done tasks by default).
- Editing `.claude/settings.json` permissions, `pyproject.toml` dependencies,
  or anything under `.claude/` in general.
- Any change to `CLAUDE.md` itself.

If you're an agent reading this and about to do one of the "needs
confirmation" items: stop, state what you're about to do and why, and wait.
