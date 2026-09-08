---
name: debug-assistant
description: Walks a structured debugging workflow -- reproduce, isolate, form a hypothesis, verify with the smallest possible change, confirm the fix, then explain the root cause -- instead of guessing at a fix or pattern-matching to a plausible-looking change. Use when asked to debug an issue, fix a bug, investigate an error or stack trace, or figure out why something is failing or behaving unexpectedly.
---

# Debug assistant

Do not jump to a fix. The point of this workflow is that a fix produced
without understanding the root cause frequently doesn't fix anything, or
fixes the symptom while leaving the actual bug in place for the next
person. Work through the steps in order and don't skip ahead.

Bound your effort: if step 3 (hypothesize) produces three hypotheses in a
row that step 4 disproves, stop iterating blindly. Step back, re-read the
actual error and surrounding code from scratch, and consider that the
mental model of the system — not just the guess — is wrong. Report what
you've ruled out rather than trying a fourth guess.

## 1. Reproduce

- Get an exact, minimal repro: the specific command, input, or request
  that triggers the failure, and the exact error/output you get. "It's
  broken" is not a repro; "running `X` with input `Y` prints traceback `Z`"
  is.
- Determine whether it's deterministic or intermittent. If intermittent,
  note what varies between runs (timing, ordering, external state,
  randomness) — that's already a clue about the category of bug.
- If you can't reproduce it at all, say so explicitly and ask for more
  information (exact steps, environment, input data) rather than guessing
  at a fix for a bug you haven't seen happen.

## 2. Isolate

- Narrow the failure to the smallest code path that still reproduces it:
  comment out or bypass unrelated code, reduce the input to the smallest
  case that still fails, binary-search across recent commits if the bug is
  new (`git bisect`, or manually checking out older versions) to find what
  introduced it.
- Distinguish an environment/configuration problem (missing dependency,
  wrong version, bad env var, stale cache) from an actual code bug before
  going further — check versions and config first if the symptom looks at
  all environmental (works on one machine, not another; worked yesterday,
  not today).
- Once isolated, identify exactly which function or line the failure
  traces to. If it's a stack trace, read the whole trace, not just the top
  frame — the top frame is often where the failure *surfaced*, not where
  it originated (e.g. a `None` that should never be `None` reaching a
  `.attribute` access three calls downstream of where it was actually set).

## 3. Form a hypothesis

- State one specific, falsifiable hypothesis about the root cause before
  touching any code: "I think X happens because Y" — not "let me try
  changing Z and see if it helps."
- Ground the hypothesis in evidence you've already gathered (the repro,
  the isolated location, the stack trace) — not a guess pattern-matched
  from a similar-looking bug you've seen before in a different codebase.
- If you have multiple plausible hypotheses, rank them by how well they
  explain *all* the observed symptoms, not just the most obvious one —
  and test the most likely one first.

## 4. Verify with the smallest possible change

- Test the hypothesis with the minimum intervention that would confirm or
  refute it: a print/log statement, a debugger breakpoint, a temporary
  assertion, or the smallest possible code change — not a rewrite of the
  surrounding logic.
- If the hypothesis is confirmed, the *fix* should also be minimal and
  should target the actual cause identified, not just make the immediate
  symptom disappear (e.g. don't wrap a crashing call in a broad
  `try/except` that swallows the error if the real problem is that it's
  being called with bad data it should never receive).
- If the hypothesis is refuted, discard it explicitly and go back to step
  3 with the new information — don't keep the disproven assumption around
  while trying the next idea.

## 5. Confirm

- Re-run the original repro from step 1 and confirm the exact original
  failure no longer occurs.
- Run the broader test suite (not just a test for this one bug) to check
  the fix didn't break something else.
- If possible, write a regression test that fails on the pre-fix code and
  passes on the post-fix code — this is the strongest evidence the fix is
  real and it prevents the same bug from coming back silently.

## 6. Explain the root cause

Do not stop at "fixed it." Report, briefly:

- **What was actually wrong** — the specific incorrect assumption, missing
  check, or logic error, not a restatement of the symptom.
- **Why it manifested the way it did** — why this particular input/timing/
  condition triggered it and others didn't.
- **Why the fix addresses the cause, not just the symptom** — if the fix
  is "added a null check," say why the value could be null in the first
  place and why checking it there (rather than fixing where it became null)
  is the right layer for the fix.
- **What was ruled out along the way**, if anything — a wrong hypothesis
  that seemed plausible is useful information for the next person who hits
  a similar-looking bug.
