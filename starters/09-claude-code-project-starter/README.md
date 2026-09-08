# 09 — Claude Code Project Starter

A template repository showing how to structure a real project so Claude Code
(or any coding agent) can work in it effectively — plus a small, real CLI
project to see it work on.

Most "how to use Claude Code" advice is abstract. This starter is not: it
ships an actual working project (`todo_cli`, a SQLite-backed todo list CLI),
a `CLAUDE.md` written specifically for that project, an example scoped
`.claude/settings.json`, and a worked example (`docs/development-workflow.md`)
of a real task — add a `--priority` flag — actually delegated, implemented,
tested, and reviewed against this exact code. Nothing in this starter calls
the Claude API or needs a key; it's entirely about project *structure* for
agentic coding, not about building an LLM feature.

## What it does

Two things, deliberately layered:

1. **`todo_cli`** — a small, genuinely useful command-line todo list manager
   with SQLite storage: `add`, `list`, `complete`, `remove`. This is the
   "project" a coding agent would be asked to work on.
2. **Claude Code scaffolding around it** — `CLAUDE.md`, `.claude/settings.json`,
   and `docs/development-workflow.md` — showing what good agent instructions,
   scoped permissions, and a delegated-task writeup look like when they're
   written against real code instead of hand-waved.

## Architecture

```
User → todo CLI (argparse) → TaskStore (sqlite3) → todo.db (SQLite file)
```

`cli.py` parses arguments and formats output; `storage.py` owns the schema
and all reads/writes; nothing else talks to SQLite directly. See
`CLAUDE.md`'s Architecture section for the exact file layout and the
reasoning behind the split.

## When to use it

Use this as a starting point when you're setting up a **new** repository (or
retrofitting an existing one) for real, sustained work with Claude Code and
want a concrete example of `CLAUDE.md` content, permission scoping, and a
delegation writeup — not just a blank template.

**Don't** use this if you want a production todo app — `todo_cli` is
intentionally minimal (no due dates, tags, sync, or multi-user support) so
it stays small enough to read in five minutes and delegate real tasks
against. It's a demo target, not a product.

## Folder structure

```
starters/09-claude-code-project-starter/
├── README.md                    # this file
├── CLAUDE.md                    # agent instructions for this repo
├── pyproject.toml                # hatchling build for todo_cli
├── .env.example                  # documents the one optional env var
├── .claude/
│   └── settings.json             # example scoped tool permissions
├── docs/
│   └── development-workflow.md   # worked example: a real delegated task
├── src/todo_cli/
│   ├── __init__.py
│   ├── __main__.py                # enables `python -m todo_cli`
│   ├── cli.py                     # argparse subcommands + formatting
│   └── storage.py                 # SQLite-backed TaskStore
└── tests/
    ├── test_storage.py            # TaskStore, via tmp_path databases
    └── test_cli.py                # main(argv), via capsys
```

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- No API key of any kind — `todo_cli` makes no network calls.

## Setup

```bash
cd starters/09-claude-code-project-starter
uv venv --python 3.13 .venv
uv pip install -e ".[dev]" --python .venv
```

(Substitute your own Python 3.11+ if you don't have 3.13; or drop the
`--python .venv` flags and use a normal virtualenv / `pip install -e ".[dev]"`.)

## Environment variables

| Name | Required? | Default | What it's for |
|---|---|---|---|
| `TODO_DB_PATH` | No | `~/.todo_cli/todo.db` | Overrides where the SQLite database file lives. `--db <path>` on the CLI takes precedence over this. |

There is no `ANTHROPIC_API_KEY` or any other credential here — this starter
is about repository structure for an agent, not about calling a model.

## Run it

```bash
# lint, format check, tests
uv run --python .venv ruff check .
uv run --python .venv ruff format --check .
uv run --python .venv pytest -q

# the CLI itself
uv run --python .venv todo add "Buy milk"
uv run --python .venv todo add "Write CLAUDE.md example"
uv run --python .venv todo list
uv run --python .venv todo complete 1
uv run --python .venv todo list --pending
uv run --python .venv todo remove 2
uv run --python .venv todo list

# also runnable as a module
uv run --python .venv python -m todo_cli list
```

Every command above was run in this directory against this exact code
before this README was written.

## Example input

```bash
uv run --python .venv todo add "Buy milk"
uv run --python .venv todo add "Write CLAUDE.md example"
uv run --python .venv todo list
uv run --python .venv todo complete 1
uv run --python .venv todo list --pending
uv run --python .venv todo remove 2
uv run --python .venv todo list
```

## Expected output

Trimmed, real output from the sequence above:

```
Added task 1: Buy milk
Added task 2: Write CLAUDE.md example
[ ] 1: Buy milk
[ ] 2: Write CLAUDE.md example
Completed task 1: Buy milk
[ ] 2: Write CLAUDE.md example
Removed task 2
[x] 1: Buy milk
```

And the test run:

```
uv run --python .venv pytest -q
...............                                                          [100%]
15 passed in 0.06s
```

## How the flow works

- **`CLAUDE.md`** is written *for this specific repo* — real file paths,
  real commands that work, an explicit "safe delegation" section splitting
  what an agent can do unprompted (add tests, add a backward-compatible
  flag, fix lint) from what needs a human's go-ahead first (schema changes
  that aren't purely additive, changing existing default behavior, editing
  its own permissions). That split is the part worth copying into your own
  project; the exact rules should change per-repo.
- **`.claude/settings.json`** shows one way to scope tool permissions:
  broad `allow` for read-only exploration and the project's own build/test
  commands, `ask` for anything that touches its own config or is
  destructive-adjacent (`rm`), `deny` for things that should never happen
  from this repo (`git push`, arbitrary `curl`, reading `.env` files). This
  file is validated as syntactically correct JSON but has not been
  exercised against a live Claude Code session — treat it as a documented
  pattern to adapt, not a certified config.
- **`docs/development-workflow.md`** is the part most worth reading closely:
  a real task (add `--priority` to `todo add`) was actually delegated,
  implemented, and tested against this code, with the genuine diff and
  genuine `pytest` output included. The change was then reverted so this
  starter ships the plain four-command CLI described above — the doc tells
  you exactly how to reapply it if you want to see it for yourself.

## Extension ideas

- Add `due_date` (nullable `TEXT`, ISO 8601) and a `--due` flag, sorting
  `list` by due date.
- Add `todo edit <id> <new description>`.
- Swap the hand-rolled `argparse` subcommands for `click` or `typer` if the
  CLI surface grows — deliberately not done here to keep the dependency
  count at zero.
- Add a `--json` output mode for scripting.

## Limitations

- `todo_cli` has no due dates, tags, priorities (in the shipped baseline),
  sync, or multi-device support — it's a teaching example, not a product.
- No schema migration tooling. The `CREATE TABLE IF NOT EXISTS` approach
  only tolerates additive schema changes; anything else needs a manual
  migration, which `CLAUDE.md` explicitly flags as needing human sign-off.
- `.claude/settings.json` is a documented pattern, validated as JSON, but
  not verified against a live Claude Code permission-prompt session — your
  actual tool names and needed patterns may differ by Claude Code version.
- Single-user, single-machine only (SQLite file on local disk).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `todo: command not found` | Package not installed / venv not active | Run `uv pip install -e ".[dev]" --python .venv`, then prefix commands with `uv run --python .venv`, or activate the venv. |
| `sqlite3.OperationalError: unable to open database file` | Parent directory for `--db` / `TODO_DB_PATH` doesn't exist and isn't creatable | `TaskStore.__init__` calls `mkdir(parents=True, exist_ok=True)` on the parent — check you have write permission to that path. |
| Tasks from a previous run "disappear" | Different `--db` path used than before, or `TODO_DB_PATH` changed | `todo` always reports which behavior it's using implicitly via `--db`; pass the same path each time, or unset/set `TODO_DB_PATH` consistently. |
| `ruff format --check .` fails | Code not run through the formatter | `uv run --python .venv ruff format .`, then re-check. |

## Production hardening

This is an educational starter, not a production system. Before relying on
it beyond a personal single-machine tool: add input length limits on task
descriptions, consider a proper schema migration tool (e.g. `alembic`-style
versioned migrations) instead of `CREATE TABLE IF NOT EXISTS`, add
concurrent-access handling if multiple processes might write to the same
`todo.db` at once (SQLite handles this reasonably but not infinitely well
under contention), and add structured logging if this ever runs
unattended. None of that is implemented here on purpose — it would obscure
the part this starter is actually teaching.
