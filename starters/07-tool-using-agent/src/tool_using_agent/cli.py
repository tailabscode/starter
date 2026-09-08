"""Command-line interface: `tool-using-agent chat "..."` / `list-tools`.

Also runnable as `python -m tool_using_agent ...` (see `__main__.py`).
"""

import argparse
import sys

from .agent import run_agent
from .config import MissingCredentialsError, load_config
from .llm import get_client
from .logging_setup import setup_logging
from .tools import build_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tool-using-agent",
        description="A bounded, tool-calling agent loop with a calculator, unit "
        "converter, local fact lookup, and a synthetic weather adapter.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force the deterministic offline stub client. Never touches the network.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the loop's max_steps bound (default: AGENT_MAX_STEPS env, or 8).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="Send one message to the agent.")
    chat_parser.add_argument("message", help="The user message to send.")

    subparsers.add_parser("list-tools", help="List the tools available to the agent.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    setup_logging(config.log_level)
    registry = build_registry()

    if args.command == "list-tools":
        for spec in registry.values():
            print(f"{spec.name}: {spec.schema['description']}")
        return 0

    # args.command == "chat"
    try:
        client = get_client(config, offline=args.offline)
    except MissingCredentialsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    max_steps = args.max_steps or config.max_steps
    try:
        result = run_agent(
            client=client, tools=registry, user_message=args.message, max_steps=max_steps
        )
    except MissingCredentialsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for call in result.tool_calls:
        status = "ERROR" if call.is_error else "ok"
        print(f"[tool] {call.name}({call.input}) -> {status}: {call.output}", file=sys.stderr)

    print(result.final_text)
    return 1 if result.hit_max_steps else 0


if __name__ == "__main__":
    raise SystemExit(main())
