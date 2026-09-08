---
name: code-review-checklist
description: Walks through a concrete, opinionated code review checklist covering correctness, security basics, test coverage, and simplicity, then reports findings in a structured, severity-ranked format. Use when asked to review a diff, a pull request, or a specific set of changed files.
---

# Code review checklist

You are reviewing a change, not admiring it. Work through every section
below against the actual diff. Skip a section only if it is genuinely not
applicable (e.g. "Security basics" for a change that touches only comments)
and say so explicitly in the report rather than silently omitting it.

Before starting, make sure you know: what the diff actually is (the changed
files and lines, not the whole repo), and what the change claims to do (PR
description, commit message, or the user's own description of the task).
If the claim and the diff don't match, that is itself a finding.

## 1. Correctness

- Does the diff do what it claims to do? Re-derive this from the code, not
  from the description.
- Off-by-one errors: loop bounds, slice indices, `<` vs `<=`.
- Null/None/empty handling: every new code path that reads a value derived
  from external input, a collection, or an optional field — what happens
  when it's `None`, empty, or the wrong type?
- Error paths: every new `try`/`except`, error return, or early exit — is
  the error actually handled, or just silenced? Is a broad
  `except Exception` hiding something that should fail loudly?
- Boundary conditions: zero, negative numbers, empty collections, maximum
  sizes, first/last element of a sequence.
- Resource cleanup: file handles, network connections, locks, database
  transactions — are they closed/released on every exit path, including
  exceptions? (Context managers used correctly?)
- Concurrency: if the change touches shared state and could run
  concurrently, is there a race? Is a lock held for the right scope (not
  too narrow, not too broad)?
- Does the diff size match the claimed task size? A "fix a typo" that
  touches twelve files, or a "small refactor" with new behavior mixed in,
  is a finding on its own — flag it even if every individual line looks fine.

## 2. Security basics

- No secrets, API keys, tokens, or credentials committed — including in
  test fixtures, example configs, or comments.
- No secrets logged or printed, even at debug level.
- Any SQL is parameterized (`?` / `%s` placeholders), never built by string
  concatenation or f-string with untrusted input.
- No `eval`, `exec`, `pickle.load`, or `yaml.load` (without
  `SafeLoader`/`safe_load`) on data that could come from outside the
  process.
- Any file path built from user/external input is validated against path
  traversal (`../`, absolute paths escaping an intended root) before use.
- Any new dependency is actually necessary — an unreviewed third-party
  package is itself a supply-chain risk; flag additions that could have
  been a few lines of stdlib instead.
- Any new network call or subprocess invocation: is the target
  fixed/trusted, or could it be influenced by external input (SSRF,
  command injection)?

## 3. Test coverage

- Every new branch, error path, and edge case identified in section 1 has
  a corresponding test — not just the happy path.
- Tests assert something meaningful. `assert True`, a test with no
  assertion, or a test that only checks "it didn't throw" is not coverage.
- Tests are deterministic: no reliance on wall-clock time, network access,
  external services, or unseeded randomness. A test that occasionally
  fails is worse than no test, because it teaches people to ignore red CI.
- Tests are independent: no shared mutable fixtures that leak state
  between tests, no dependency on test execution order.
- If a bug is being fixed, is there a regression test that would have
  caught it — i.e. does it fail against the pre-fix code and pass against
  the post-fix code?

## 4. Simplicity

- No speculative abstraction: no new base class, plugin system, config
  option, or "flexibility" that isn't used by the current call site.
- No duplicated logic that already exists elsewhere in the file/module and
  could be reused instead of copy-pasted.
- Function/file length reasonable for what it does — a function that grew
  past ~50 lines or a file past ~250 lines in this diff is worth asking
  "should this be split?"
- Naming matches the existing conventions in the surrounding code, not a
  new convention introduced just for this change.
- Would a senior engineer on this specific codebase look at this diff and
  say "this is more complicated than it needs to be"? If yes, say which
  part and what the simpler version would look like.

## How to report findings

Structure the report as:

```
## Summary
<one or two sentences: does this diff do what it says, and is it ready?>

## Blockers
- <file:line> — <what's wrong, in one sentence> — <why it matters>

## Should fix
- <file:line> — <what's wrong> — <why>

## Nits
- <file:line> — <minor, non-blocking>

## Verdict
Approve / Approve with comments / Request changes
```

Rules for the report:

- Every finding cites a specific file and line (or line range) — no vague
  "the error handling could be better" without pointing at the code.
- Severity is honest: a real correctness bug or security issue is a
  **blocker**, not a nit. Style preferences with no functional impact are
  **nits**, not blockers.
- If there are zero findings in a section, say so briefly ("Security
  basics: no external input or secrets touched by this diff") rather than
  omitting the section — that tells the reader you checked, not skipped it.
- Don't pad the report with praise for things that are merely correct.
  Correctness is the baseline, not a finding.
