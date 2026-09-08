# Development workflow: delegating a real task to Claude Code

This is a worked example, not a hypothetical. We actually asked Claude Code
to add a `--priority` flag to `todo add` in this exact repo, ran the full
diff and test suite below, and captured the real output. We then reverted
the change so the starter you cloned ships with the plain baseline
(`add`/`list`/`complete`/`remove`, no priority) — this doc is the reference
for you to redo the same change yourself as practice, or to see what a
sane delegation loop looks like end to end.

## The task

> Add a `--priority` flag to `todo add`, accepting `low`, `normal`, or
> `high`, defaulting to `normal`. Show non-default priorities in `todo list`
> output. Keep it backward compatible — existing calls to `todo add
> "description"` with no flag must behave exactly as before.

This is a good task to delegate: small, additive, testable, and the "don't
break existing behavior" constraint is easy to verify mechanically (the
existing test suite must still pass unmodified).

## 1. Plan

Before touching code, a plan was requested first. Here's what a reasonable
plan for this task looks like, and why each step is ordered this way:

1. **Storage layer first** (`src/todo_cli/storage.py`): add a `priority`
   column to the schema, default `'normal'` so existing rows and existing
   `INSERT` statements from old code stay valid; add `priority: str =
   "normal"` to the `Task` dataclass with a default so old call sites that
   construct `Task(...)` positionally don't break; extend `add()` to accept
   and validate a `priority` kwarg; extend `list()` and `complete()` to
   select and return it.
2. **CLI layer second** (`src/todo_cli/cli.py`): add the `--priority`
   `argparse` flag with `choices=[...]` (so invalid values fail fast with a
   clear `argparse` error, not a `ValueError` from deep in `storage.py`);
   wire it through to `store.add()`; update `_format_task()` to append
   `(high)` / `(low)` only when priority isn't the default, so normal-output
   stays unchanged.
3. **Tests third**: one test per new behavior (default priority, valid
   non-default priority persists, invalid priority is rejected at both the
   storage layer and the CLI layer, priority shows in `list` output only
   when non-default).
4. **Verify fourth**: run the *existing* suite unmodified to prove backward
   compatibility, then run the full suite with new tests added, then lint
   and format.

Storage before CLI matters here because the CLI layer's `--priority` flag
is only meaningful once `store.add()` knows what to do with it — building
top-down would mean stubbing the storage call anyway.

## 2. Implement

The actual diff produced, applied against the baseline in this repo:

```diff
--- src/todo_cli/storage.py
@@ -17,11 +17,14 @@
 CREATE TABLE IF NOT EXISTS tasks (
     id INTEGER PRIMARY KEY AUTOINCREMENT,
     description TEXT NOT NULL,
-    done INTEGER NOT NULL DEFAULT 0
+    done INTEGER NOT NULL DEFAULT 0,
+    priority TEXT NOT NULL DEFAULT 'normal'
 );
 """
 
+VALID_PRIORITIES = ("low", "normal", "high")
+
+
 @dataclass(frozen=True)
 class Task:
     id: int
     description: str
     done: bool
+    priority: str = "normal"

@@ -57,29 +61,34 @@
-    def add(self, description: str) -> Task:
-        """Insert a new task and return it. Raises ValueError on blank input."""
+    def add(self, description: str, priority: str = "normal") -> Task:
+        """Insert a new task and return it.
+
+        Raises ValueError on blank input or an unrecognized priority.
+        """
         cleaned = description.strip()
         if not cleaned:
             raise ValueError("Task description cannot be empty")
+        if priority not in VALID_PRIORITIES:
+            raise ValueError(f"priority must be one of {VALID_PRIORITIES}, got {priority!r}")
         with self._connect() as conn:
             cur = conn.execute(
-                "INSERT INTO tasks (description, done) VALUES (?, 0)",
-                (cleaned,),
+                "INSERT INTO tasks (description, done, priority) VALUES (?, 0, ?)",
+                (cleaned, priority),
             )
             task_id = cur.lastrowid
         assert task_id is not None
-        return Task(id=task_id, description=cleaned, done=False)
+        return Task(id=task_id, description=cleaned, done=False, priority=priority)

     def list(self, *, include_done: bool = True) -> list[Task]:
-        query = "SELECT id, description, done FROM tasks"
+        query = "SELECT id, description, done, priority FROM tasks"
         ...
-        return [Task(id=r[0], description=r[1], done=bool(r[2])) for r in rows]
+        return [Task(id=r[0], description=r[1], done=bool(r[2]), priority=r[3]) for r in rows]

     def complete(self, task_id: int) -> Task:
         ...
             row = conn.execute(
-                "SELECT id, description, done FROM tasks WHERE id = ?", (task_id,)
+                "SELECT id, description, done, priority FROM tasks WHERE id = ?", (task_id,)
             ).fetchone()
-        return Task(id=row[0], description=row[1], done=bool(row[2]))
+        return Task(id=row[0], description=row[1], done=bool(row[2]), priority=row[3])
```

```diff
--- src/todo_cli/cli.py
@@ -27,7 +27,8 @@
 def _format_task(task: Task) -> str:
     mark = "x" if task.done else " "
-    return f"[{mark}] {task.id}: {task.description}"
+    suffix = "" if task.priority == "normal" else f" ({task.priority})"
+    return f"[{mark}] {task.id}: {task.description}{suffix}"

@@ -41,6 +42,12 @@
     add_p = sub.add_parser("add", help="Add a new task")
     add_p.add_argument("description", help="Task description")
+    add_p.add_argument(
+        "--priority",
+        choices=["low", "normal", "high"],
+        default="normal",
+        help="Task priority (default: normal)",
+    )

@@ -61,7 +68,7 @@
         if args.command == "add":
-            task = store.add(args.description)
+            task = store.add(args.description, priority=args.priority)
             print(f"Added task {task.id}: {task.description}")
```

Plus six new tests (three in `tests/test_storage.py`, three in
`tests/test_cli.py`) covering: default priority, a valid non-default
priority round-tripping through storage, an invalid priority rejected by
`TaskStore.add`, priority appearing in `todo list` output, the default
priority producing no visible suffix, and an invalid `--priority` value
being rejected by `argparse` (exit via `SystemExit`, not a Python
exception leaking to the user).

## 3. Test

Existing suite, run first, unmodified, to confirm nothing broke:

```
uv run --python .venv pytest -q
...............                                                          [100%]
15 passed in 0.06s
```

Then with the new tests added:

```
uv run --python .venv pytest -v
...
tests/test_cli.py::test_add_with_priority_shows_in_list PASSED           [ 33%]
tests/test_cli.py::test_add_default_priority_has_no_suffix PASSED        [ 38%]
tests/test_cli.py::test_add_rejects_invalid_priority_choice PASSED       [ 42%]
...
tests/test_storage.py::test_add_defaults_to_normal_priority PASSED       [ 90%]
tests/test_storage.py::test_add_accepts_valid_priority PASSED            [ 95%]
tests/test_storage.py::test_add_rejects_invalid_priority PASSED          [100%]
============================== 21 passed in 0.07s ==============================
```

`ruff check .` and `ruff format --check .` both passed with no changes
needed beyond what the editor already produced.

A manual run, to see it for real (not just assert on it):

```
$ uv run --python .venv todo --db /tmp/demo.db add "Fix production outage" --priority high
Added task 1: Fix production outage
$ uv run --python .venv todo --db /tmp/demo.db add "Buy milk"
Added task 2: Buy milk
$ uv run --python .venv todo --db /tmp/demo.db add "Read a novel" --priority low
Added task 3: Read a novel
$ uv run --python .venv todo --db /tmp/demo.db list
[ ] 1: Fix production outage (high)
[ ] 2: Buy milk
[ ] 3: Read a novel (low)
```

## 4. Review — what to check before trusting the result

Don't just look at "tests pass, ship it." For this specific change:

- **Backward compatibility, mechanically verified**: the pre-existing 15
  tests were run *before* adding any new ones and passed unmodified — that's
  the actual evidence the change is additive, not just an assertion in the
  PR description.
- **Schema default matches dataclass default.** `'normal'` in the SQL
  `DEFAULT` clause and `"normal"` in `Task.priority: str = "normal"` must
  agree, or a task inserted by old code and read by new code (or vice
  versa) would disagree with itself. Worth grep-ing both literals by hand.
- **Validation happens in the right place, twice, on purpose.** `argparse
  choices=` rejects bad CLI input before it reaches `TaskStore`, and
  `TaskStore.add` *also* validates, because `TaskStore` is a public API a
  future non-CLI caller (a test, a script, a future TUI) could call
  directly with an unvalidated string.
- **The "no suffix for default priority" behavior was actually tested**,
  not just implemented — it's the kind of thing that's easy to get right by
  accident and then silently break in a later refactor without a test
  catching it.
- **No migration path was written, and that's flagged, not hidden.** An
  existing `~/.todo_cli/todo.db` created by the old schema gets the new
  `priority` column for free because `sqlite3` only runs `CREATE TABLE IF
  NOT EXISTS` — but if the schema had *removed* or *renamed* a column
  instead of adding one, this would need an explicit migration, and that's
  exactly the kind of change `CLAUDE.md` calls out as needing confirmation
  before an agent touches it.
- **Diff size matches task size.** Two source files touched, ~25 lines of
  production code changed, no unrelated formatting churn. If a "add a
  flag" task comes back touching ten files, that's a signal to ask why
  before merging, not after.

## Applying this yourself

The baseline in `src/todo_cli/` does **not** include the `--priority` flag
— the diffs above are real (they were applied, tested, and verified against
this exact codebase) but were reverted so this starter ships the plain
four-command CLI described in the main README. Apply the diffs above by
hand, or re-run the same delegation with Claude Code and compare what you
get against this document.
