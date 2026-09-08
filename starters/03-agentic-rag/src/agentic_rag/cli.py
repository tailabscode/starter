"""Command-line entry point: ``agentic-rag query|plan|trace``."""

from __future__ import annotations

import argparse
import json
import sys

from agentic_rag.agent import run_agent
from agentic_rag.config import load_config
from agentic_rag.errors import MissingCredentialsError
from agentic_rag.llm import get_client
from agentic_rag.logging_setup import configure_logging
from agentic_rag.retrieval import HybridRetriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-rag", description="An agent that decides whether and how to retrieve."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    query = sub.add_parser("query", help="Ask a question; runs the plan-then-act loop.")
    query.add_argument("question")
    query.add_argument(
        "--offline", action="store_true", help="Force the deterministic stub client."
    )
    query.add_argument("--max-steps", type=int, default=None)

    plan = sub.add_parser("plan", help="Show the query plan only; makes no tool calls.")
    plan.add_argument("question")
    plan.add_argument("--offline", action="store_true")

    sub.add_parser("trace", help="Pretty-print the trace from the last `query` run.")

    return parser


def _run_query(args: argparse.Namespace) -> int:
    config = load_config()
    logger = configure_logging()
    retriever = HybridRetriever.from_data_dir(config.data_dir)
    try:
        llm = get_client(config, offline=args.offline)
    except MissingCredentialsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    max_steps = args.max_steps or config.max_steps
    result = run_agent(args.question, llm=llm, retriever=retriever, max_steps=max_steps)

    print(result.answer)
    if result.citations:
        print(f"\nCitations: {', '.join(result.citations)}")
    config.trace_path.parent.mkdir(parents=True, exist_ok=True)
    config.trace_path.write_text(json.dumps(result.trace.to_dict(), indent=2), encoding="utf-8")
    logger.info("Trace written.", extra={"trace_path": str(config.trace_path)})
    return 0


def _run_plan(args: argparse.Namespace) -> int:
    config = load_config()
    retriever = HybridRetriever.from_data_dir(config.data_dir)
    try:
        llm = get_client(config, offline=args.offline)
    except MissingCredentialsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    plan = llm.plan_query(args.question, retriever.list_topics())
    print(json.dumps(plan.model_dump(), indent=2))
    return 0


def _run_trace(_args: argparse.Namespace) -> int:
    config = load_config()
    if not config.trace_path.exists():
        print('No trace found yet. Run `agentic-rag query "..."` first.')
        return 1
    print(config.trace_path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {"query": _run_query, "plan": _run_plan, "trace": _run_trace}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
