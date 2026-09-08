"""Command-line interface: `agent-evals run` / `compare` / `show`.

Also runnable as `python -m agent_evals ...` (see `__main__.py`).
"""

import argparse
import sys
from pathlib import Path

from .compare import CompareResult, load_and_compare
from .config import MissingCredentialsError, load_config
from .dataset import DEFAULT_DATASET_PATH
from .llm import get_client
from .logging_setup import setup_logging
from .runner import DEFAULT_THRESHOLD, RunSummary, run_eval
from .tracing import DEFAULT_RUNS_DIR, TraceRecord, read_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-evals",
        description="A deterministic eval runner with keyword coverage, tool-use "
        "correctness, and an (optional) LLM-judge groundedness metric.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force the deterministic offline stub client for both the target "
        "agent and the LLM-judge metric. Never touches the network.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the eval suite and write a trace.")
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum aggregate pass rate to exit 0 (default: {DEFAULT_THRESHOLD}).",
    )
    run_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to a JSONL eval dataset."
    )
    run_parser.add_argument("--run-id", default=None, help="Override the generated run id.")

    compare_parser = subparsers.add_parser(
        "compare", help="Diff two runs case-by-case; flag regressions and improvements."
    )
    compare_parser.add_argument("run_a", help="Earlier run id (baseline).")
    compare_parser.add_argument("run_b", help="Later run id (candidate).")

    show_parser = subparsers.add_parser("show", help="Print the full trace for one run.")
    show_parser.add_argument("run_id")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    setup_logging(config.log_level)

    if args.command == "run":
        try:
            client = get_client(config, offline=args.offline)
        except MissingCredentialsError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        try:
            summary = run_eval(args.dataset, client, DEFAULT_RUNS_DIR, run_id=args.run_id)
        except (MissingCredentialsError, RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        _print_run_summary(summary)
        gate_passed = summary.pass_rate >= args.threshold
        print(
            f"\npass_rate={summary.pass_rate:.2f} threshold={args.threshold:.2f} "
            f"-> {'PASS' if gate_passed else 'FAIL'}"
        )
        return 0 if gate_passed else 1

    if args.command == "compare":
        try:
            result = load_and_compare(DEFAULT_RUNS_DIR, args.run_a, args.run_b)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        _print_compare(result)
        return 0

    if args.command == "show":
        try:
            records = read_trace(DEFAULT_RUNS_DIR / f"{args.run_id}.jsonl")
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        _print_records(records)
        return 0

    return 1


def _print_run_summary(summary: RunSummary) -> None:
    print(f"run_id={summary.run_id}")
    for r in summary.records:
        status = "PASS" if r.passed else "FAIL"
        groundedness = (
            f" groundedness={r.groundedness_llm_score:.2f}"
            if r.groundedness_llm_score is not None
            else ""
        )
        print(
            f"[{status}] {r.case_id} ({r.category}) "
            f"kw={r.keyword_coverage_score:.2f} tool={r.tool_use_score:.2f}{groundedness} "
            f"latency_ms={r.latency_ms:.1f}"
        )
    print(f"\n{summary.passed}/{summary.total} cases passed")


def _print_compare(result: CompareResult) -> None:
    print(f"comparing {result.run_id_a} -> {result.run_id_b}")
    changed = [d for d in result.diffs if d.status != "unchanged"]
    if not changed:
        print("no changes")
    for diff in changed:
        print(
            f"[{diff.status}] {diff.case_id} ({diff.category}): {diff.passed_a} -> {diff.passed_b}"
        )
    unchanged = len(result.diffs) - len(result.regressions) - len(result.improvements)
    print(
        f"\n{len(result.regressions)} regression(s), {len(result.improvements)} improvement(s), "
        f"{unchanged} unchanged"
    )


def _print_records(records: list[TraceRecord]) -> None:
    for r in records:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.case_id} ({r.category})")
        print(f"  input:  {r.input}")
        print(f"  output: {r.output}")
        if r.tool_calls:
            calls = ", ".join(f"{c.name}({c.input})" for c in r.tool_calls)
            print(f"  tool_calls: {calls}")
        print(
            f"  scores: keyword_coverage={r.keyword_coverage_score:.2f} "
            f"tool_use={r.tool_use_score:.2f} groundedness_llm={r.groundedness_llm_score}"
        )
        print(f"  latency_ms={r.latency_ms:.1f} timestamp={r.timestamp}")


if __name__ == "__main__":
    raise SystemExit(main())
