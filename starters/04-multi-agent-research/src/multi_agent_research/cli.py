"""Command-line entry point: ``multi-agent-research research|show|list``."""

from __future__ import annotations

import argparse
import sys

from multi_agent_research.config import load_config
from multi_agent_research.corpus import Corpus
from multi_agent_research.errors import (
    CorpusNotFoundError,
    MissingCredentialsError,
    RunNotFoundError,
)
from multi_agent_research.llm import get_client
from multi_agent_research.logging_setup import configure_logging
from multi_agent_research.orchestrator import run_research
from multi_agent_research.state import ResearchState


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-agent-research",
        description=(
            "A coordinator/researcher/critic/synthesiser workflow with concurrent research "
            "and a bounded critique-and-revise loop."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    research = sub.add_parser("research", help="Run the full pipeline on a question.")
    research.add_argument("question")
    research.add_argument(
        "--offline", action="store_true", help="Force the deterministic stub client."
    )

    show = sub.add_parser("show", help="Print the saved state for one run.")
    show.add_argument("run_id")

    sub.add_parser("list", help="List saved run ids, newest last.")

    return parser


def _run_research(args: argparse.Namespace) -> int:
    config = load_config()
    logger = configure_logging()
    try:
        corpus = Corpus.from_data_dir(config.data_dir)
    except CorpusNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        llm = get_client(config, offline=args.offline)
    except MissingCredentialsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    state = run_research(args.question, llm=llm, corpus=corpus, config=config)

    print(f"Run id: {state.run_id}")
    print(f"Rounds: {state.revision_round}  Termination: {state.termination_reason}")
    print()
    print(state.final_report or "(no report produced)")
    all_sources = sorted({s["doc_id"] for f in state.findings for s in f.get("sources", [])})
    print(f"\nSources used: {', '.join(all_sources) if all_sources else '(none)'}")
    logger.info(
        "Run complete.",
        extra={"run_id": state.run_id, "runs_dir": str(config.runs_dir)},
    )
    return 0


def _run_show(args: argparse.Namespace) -> int:
    config = load_config()
    try:
        state = ResearchState.load(config.runs_dir, args.run_id)
    except RunNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(state.to_json())
    return 0


def _run_list(_args: argparse.Namespace) -> int:
    config = load_config()
    run_ids = ResearchState.list_runs(config.runs_dir)
    if not run_ids:
        print('No runs found yet. Run `multi-agent-research research "..."` first.')
        return 0
    for run_id in run_ids:
        state = ResearchState.load(config.runs_dir, run_id)
        print(
            f"{run_id}  rounds={state.revision_round}  "
            f"termination={state.termination_reason}  question={state.question!r}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "research":
        return _run_research(args)
    if args.command == "show":
        return _run_show(args)
    if args.command == "list":
        return _run_list(args)
    # pragma: no cover - argparse's `required=True` on the subparsers makes this unreachable
    raise ValueError(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
