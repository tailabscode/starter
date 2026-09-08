# 10 — Claude Code Skills

Three genuinely usable Claude Code skills, ready to drop into a project.

A "skill" here is a packaged, reusable set of instructions — a markdown
file with a bit of YAML metadata — that tells Claude Code exactly how to
perform a recurring task the way *you* want it done, instead of you
re-explaining your review checklist, test conventions, or debugging process
in every single session. This starter ships three: `code-review-checklist`,
`test-generator`, and `debug-assistant`, each with concrete, opinionated
instructions rather than generic advice, plus a small script that validates
every skill's frontmatter and instructions actually parse and aren't empty
stubs.

## What it does

- `skills/code-review-checklist/SKILL.md` — a specific review checklist
  (correctness, security basics, test coverage, simplicity) with a
  structured, severity-ranked reporting format.
- `skills/test-generator/SKILL.md` — a workflow for identifying a
  function's contract and edge cases, finding the repo's existing test
  conventions, and writing tests that match them.
- `skills/debug-assistant/SKILL.md` — a structured debugging loop:
  reproduce, isolate, hypothesize, verify with the smallest change,
  confirm, explain the root cause. Not "guess a fix and see if it works."
- `scripts/check_skills.py` — validates that every `SKILL.md` has correct,
  non-empty YAML frontmatter (`name`, `description`) and a non-trivial
  instructions body, using only the standard library.

## Architecture

```
your session: "review this diff"
        │
        ▼
Claude Code matches the request against each installed skill's
`description` (or you invoke one by name)
        │
        ▼
matching SKILL.md body is loaded into context as instructions
        │
        ▼
Claude Code follows those instructions for the rest of the task
```

Skills are pure markdown + YAML frontmatter — there is no runtime code, no
API key, and nothing to install as a Python package. The only code in this
starter is the validation script.

## When to use it

Use this starter as a source of three ready-to-copy skills for a real
project, and as a worked example of what a *specific, actionable* SKILL.md
looks like versus a vague one. Read `skills/*/SKILL.md` directly to adapt
the checklist/workflow content to your own team's actual standards — the
value of a skill is in how concrete it is, and your standards will differ
from the ones written here.

**Don't** expect these to be invoked automatically with zero setup outside
Claude Code, and don't expect them to replace human review on anything
security- or correctness-critical — a skill makes Claude Code's process
more consistent, it doesn't make its judgment infallible.

## Folder structure

```
starters/10-claude-code-skills/
├── README.md
├── skills/
│   ├── code-review-checklist/
│   │   └── SKILL.md
│   ├── test-generator/
│   │   └── SKILL.md
│   └── debug-assistant/
│       └── SKILL.md
└── scripts/
    └── check_skills.py     # validates frontmatter + non-empty instructions
```

## Prerequisites

- Claude Code, to actually use the skills.
- Python 3.11+ (any Python 3 works in practice; the check script has no
  version-specific syntax), only to run `scripts/check_skills.py`. No
  packages to install — it's stdlib only.

## Setup

**To validate the skills** (no installation needed):

```bash
cd starters/10-claude-code-skills
python3 scripts/check_skills.py
```

**To actually use these skills with Claude Code**, copy (or symlink) the
skill directories into a project's `.claude/skills/` directory — this is
the real, documented project-level location Claude Code auto-discovers
skills from at session start:

```bash
mkdir -p /path/to/your-project/.claude/skills
cp -r skills/code-review-checklist /path/to/your-project/.claude/skills/
cp -r skills/test-generator        /path/to/your-project/.claude/skills/
cp -r skills/debug-assistant       /path/to/your-project/.claude/skills/
```

To make a skill available to you across *every* project instead of just
one, copy it to `~/.claude/skills/<skill-name>/` (your personal, user-level
skills directory) instead of a project's `.claude/skills/`.

Either way, the mechanism is the same: a directory containing exactly one
`SKILL.md` with `name` + `description` frontmatter, named to match its
directory. There's no build step, registry, or separate install command —
copying the folder *is* the installation.

## Environment variables

None. Skills are static instructions loaded into context; they make no API
calls and read no credentials themselves. (Whatever Claude Code session
you use them in may of course need its own `ANTHROPIC_API_KEY`, but that's
unrelated to this starter.)

## Run it

```bash
# Validate all three skills (offline, no network, <1s)
python3 scripts/check_skills.py
```

Real output from this exact command, in this directory:

```
PASS  code-review-checklist
PASS  debug-assistant
PASS  test-generator

All skills valid.
```

To invoke a skill inside a Claude Code session (after copying it into
`.claude/skills/` as shown above): describe the task in terms that match
the skill's `description` (e.g. "review this diff" naturally triggers
`code-review-checklist`), or name it explicitly (e.g. "use the
test-generator skill on `src/foo.py`").

## Example input

A worked transcript-style walkthrough of `code-review-checklist`, run
against a small, deliberately flawed snippet:

```python
def get_user_by_email(conn, email):
    query = "SELECT * FROM users WHERE email = '" + email + "'"
    cursor = conn.execute(query)
    row = cursor.fetchone()
    return row
```

**Prompt:** "Review this function."

## Expected output

Following `code-review-checklist`'s own instructions and reporting format
exactly, a session working through this snippet would produce:

```
## Summary
This function builds SQL by string concatenation with unsanitized input —
a SQL injection vulnerability — and ships with no tests. Not ready as-is.

## Blockers
- snippet.py:2 — `email` is concatenated directly into the SQL string,
  so a value like `' OR '1'='1` changes the query's meaning entirely.
  Use a parameterized query instead:
  `conn.execute("SELECT * FROM users WHERE email = ?", (email,))`.

## Should fix
- snippet.py:1-5 — No tests included. At minimum: a test for a found
  user, a test for "no user with this email" (confirms the `None` return
  is intentional and handled by callers), and a regression test asserting
  a malicious `email` value does not alter query results once the fix
  above is applied.
- snippet.py:4 — Returns `None` silently when no row matches, with no
  docstring or type hint stating that. A caller that assumes a row is
  always present will hit a confusing downstream error instead of a clear
  one. Document it (`-> sqlite3.Row | None`) or raise a named exception,
  whichever matches how the rest of this codebase signals "not found."

## Nits
- snippet.py:2 — `SELECT *` pulls every column even though only some may
  be needed; naming the columns makes the function resilient to schema
  changes and makes what it actually uses explicit.

## Verdict
Request changes — the SQL injection is a blocker on its own regardless of
the other findings.
```

This is a realistic example worked through by hand against the skill's own
checklist and reporting template, not a captured transcript from a live
session — it demonstrates the shape and specificity the skill is meant to
produce.

## How the flow works

Each `SKILL.md` follows the same shape: YAML frontmatter with `name` (must
match the directory name — that's what the loader keys off, and what
`check_skills.py` verifies) and `description` (the trigger — Claude Code
matches your request's intent against this to decide when the skill is
relevant, so it needs to say specifically *when* to use the skill, not just
what it does in the abstract). The markdown body is written as direct
operating instructions ("do X, then check Y") rather than marketing copy
describing the skill — because the body is exactly what gets loaded into
the model's context and followed, not documentation shown to a human.

`scripts/check_skills.py` deliberately avoids a YAML library: this
starter's frontmatter is two flat string fields, so a stdlib
split-on-`'---'` plus `key: value` line parser is enough, and it keeps the
"tests" for this starter dependency-free — see the spec's dependency
budget rationale for why that matters even for a two-field parser.

## Extension ideas

- A fourth skill, e.g. `repo-analyzer`, that walks dependency files
  (`pyproject.toml`, `package.json`) and directory structure to produce a
  quick architecture summary for someone new to a codebase.
- A `docs-updater` skill that checks whether docstrings/README examples
  still match the code they describe after a change.
- Extend `check_skills.py` to also flag `description` fields that are too
  generic to trigger reliably (e.g. under ~10 words with no concrete
  "use when" clause).

## Limitations

- Skills are instructions, not code — they rely on the model actually
  following them; nothing here mechanically enforces that a review or test
  run happened the way the skill describes.
- `check_skills.py` validates structure (frontmatter present, body
  non-trivial) — it cannot and does not judge whether the instructions
  are *good* advice for your specific codebase.
- The worked example in this README is a hand-verified illustration of the
  skill's expected output shape, not a captured transcript from an actual
  Claude Code run — said explicitly here rather than presented as one.
- No mechanism in this starter auto-installs skills into `.claude/skills/`
  — copying the folder is a manual, one-line step by design (no hidden
  install script to trust or audit).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python3 scripts/check_skills.py` reports a name mismatch | The `name:` field in frontmatter doesn't match its directory name | Rename one to match the other — Claude Code's real skill loader keys off the directory name, so a mismatch would also break actual use, not just validation. |
| A copied skill never gets triggered automatically in a session | `description` doesn't clearly state *when* to use it, or the request doesn't semantically match it | Rephrase your request to match the description's language, or invoke the skill by name explicitly. |
| `check_skills.py` says "instructions body is only N chars" | `SKILL.md` body is a stub with no real content | Fill in real, actionable instructions — the 200-character floor exists specifically to catch stub files. |
| `No skills/ directory found` | Running the script from the wrong working directory | `cd` into `starters/10-claude-code-skills` first, or run with the full path — the script resolves `skills/` relative to its own file location either way, so this should be rare. |

## Production hardening

These are instruction files, not a deployed system — "production hardening"
mostly means process hardening: review changes to `SKILL.md` files the same
way you'd review any change to team process documentation (they directly
shape what an agent does on your codebase), keep descriptions specific
enough that skills trigger predictably rather than either never firing or
firing on unrelated requests, and periodically re-run
`scripts/check_skills.py` in CI if you maintain more than a handful of
skills, so a broken frontmatter edit is caught before someone relies on it.
