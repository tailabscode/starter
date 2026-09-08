---
name: test-generator
description: Generates tests for a given function or module by first identifying its contract and edge cases, then finding the existing test conventions in the repo, then writing tests that match those conventions. Use when asked to add tests, write tests for a specific function/module, or increase test coverage.
---

# Test generator

Do not write tests from a generic template. The goal is tests that read
like they belong in this specific repo and actually exercise the target
code's real behavior — not `assert True`, not tests that only check "it
didn't crash."

Work through these steps in order.

## 1. Read the target completely

Read the whole function or module being tested — not just the signature.
You need:

- The full parameter list, types, and defaults.
- Every branch (`if`/`elif`/`else`, early returns, loops).
- Every exception it can raise, explicitly (`raise`) or implicitly
  (indexing, dict access, type coercion, division).
- Any side effects: file writes, network calls, mutation of arguments or
  shared state, logging.
- The docstring, if present — but verify the code actually matches it;
  docstrings drift out of date. If they disagree, trust the code and note
  the discrepancy.

## 2. Identify the contract

Write down, even just mentally:

- **Inputs**: types, valid ranges, what "invalid" looks like for each
  parameter.
- **Outputs**: type and shape of the return value, and what it means.
- **Side effects**: what changes in the world when this runs (files,
  database rows, mutated objects, network requests).
- **Preconditions**: what must be true before calling it (e.g. "the file
  must exist", "the list must be non-empty").
- **Documented exceptions**: what's explicitly raised, and under what
  condition each one fires.

## 3. Enumerate edge cases systematically

Don't rely on intuition alone — check each of these categories against the
actual parameters:

- **Boundary values**: empty string/list/dict, zero, negative numbers,
  the single-element case, the maximum size the code assumes (buffer
  limits, off-by-one-prone loop bounds).
- **Type edge cases**: `None` where a value is expected, wrong type passed
  where duck-typing might silently misbehave, unicode/non-ASCII strings if
  the code does any string processing.
- **Error paths**: one test per distinct exception the function can raise,
  each with the specific input that triggers it — not a single catch-all
  "raises on bad input" test.
- **State-dependent behavior**: if behavior depends on prior calls (a
  counter, a cache, a database row created earlier), test both the
  first-call and repeated-call behavior.
- **Concurrent/idempotency behavior**: if the function is meant to be
  idempotent (safe to call twice with the same input), test that
  explicitly.

## 4. Find the existing test conventions before writing anything

Search the repo for tests near the target code (same package, or a
`tests/` directory mirroring the source layout) and note:

- **Framework**: pytest, unittest, jest, etc. — use the same one, don't
  introduce a second test framework into a repo that already has one.
- **Naming**: `test_<behavior>` vs `test_<function>_<condition>` — match
  whatever pattern dominates nearby files.
- **Fixture/setup pattern**: does the repo use pytest fixtures, a
  `setUp`/`tearDown`, factory functions, or inline construction? Reuse
  existing fixtures rather than duplicating setup code.
- **Assertion style**: plain `assert`, a custom assertion helper, matcher
  library (e.g. `pytest.approx`, `unittest.mock.assert_called_with`) —
  match what's already used for the same kind of check.
- **Mocking conventions**: does the repo mock network/filesystem calls, or
  use fakes/stubs, or run against real local resources (e.g. `tmp_path`,
  an in-memory database)? Follow the existing pattern rather than
  introducing a new one for this one test file.
- **Test data**: inline literals vs. fixture files vs. factory functions —
  match the dominant style.

If there are no existing tests to learn from, default to the simplest
idiomatic pattern for the language/framework in use, and say so explicitly
in your summary so the human reviewer knows you set the convention rather
than followed one.

## 5. Write the tests

- One test per distinct behavior — don't cram multiple unrelated
  assertions about different behaviors into one test function; when it
  fails, the name should already tell you what broke.
- Descriptive names that state the behavior, not the mechanism:
  `test_add_rejects_empty_description`, not `test_add_2`.
- Test the public contract, not private implementation details — don't
  assert on internal variable names, private helper call counts, or
  anything not part of what the function promises to callers. Tests that
  couple to implementation details break on harmless refactors.
- No `assert True`, no empty test bodies, no test that only checks "no
  exception was raised" when the function's actual job is to return a
  specific value.
- Each test is independent: no shared mutable state between tests, no
  reliance on execution order. Use fresh fixtures (e.g. `tmp_path`, a new
  instance) per test.
- No network calls, no real sleeps/timing dependence, no reliance on
  system clock or external services — mock or fake anything like that, or
  use the repo's existing pattern for it.
- Cover the edge cases from step 3 — at minimum, one test per distinct
  exception path and one test per boundary condition identified.

## 6. Verify before declaring done

- Run the new tests and confirm they pass against the current code.
- Sanity-check that each test can actually fail: mentally (or literally,
  by temporarily breaking the target code) confirm each new test would
  catch a real regression, not just execute without asserting anything
  meaningful. A test that can never fail is not a test.
- Run the full existing test suite too, not just the new file — confirm
  nothing else broke and no naming collision was introduced.
- Report which edge cases you covered and, just as importantly, which
  ones you deliberately left out and why (e.g. "did not test concurrent
  access — the function isn't documented as thread-safe").
