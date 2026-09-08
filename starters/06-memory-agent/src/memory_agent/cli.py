"""Command-line interface: chat, memories, forget, demo."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .chat import run_turn
from .config import load_config
from .embedder import HashingEmbedder
from .errors import MemoryAgentError
from .llm import get_client
from .logging_setup import configure_logging
from .session import SessionMemory
from .store import MemoryStore


def _cmd_chat(args: argparse.Namespace) -> int:
    config = load_config()
    embedder = HashingEmbedder(dim=config.embedding_dim)
    llm_client = get_client(config, force_offline=args.offline)

    with MemoryStore(config.db_path, embedder) as store:
        session = SessionMemory(session_id=args.session_id)
        result = run_turn(
            session,
            store,
            llm_client,
            args.message,
            top_k=config.retrieval_top_k,
            dedup_threshold=config.dedup_threshold,
            max_memories_per_turn=config.max_memories_per_turn,
        )

    print(f"session_id={args.session_id}")
    print(f"you: {args.message}")
    if result.retrieved:
        for record, score in result.retrieved:
            print(f"remembered: ({record.category}, score={score:.2f}) {record.content}")
    else:
        print("remembered: (nothing remembered yet about this user)")
    print(f"assistant: {result.reply}")
    for record, action_taken in result.memory_writes:
        print(f"memory: {action_taken} ({record.category}) {record.content} [id={record.id}]")
    return 0


def _cmd_memories(args: argparse.Namespace) -> int:
    config = load_config()
    embedder = HashingEmbedder(dim=config.embedding_dim)
    with MemoryStore(config.db_path, embedder) as store:
        records = store.list_all()
        if not records:
            print("no memories stored yet -- run `chat` first")
            return 0
        for record in records:
            print(
                f"{record.id}  [{record.category}]  {record.content}  "
                f"(created={record.created_at}, last_accessed={record.last_accessed_at}, "
                f"session={record.source_session_id})"
            )
    return 0


def _cmd_forget(args: argparse.Namespace) -> int:
    config = load_config()
    embedder = HashingEmbedder(dim=config.embedding_dim)
    with MemoryStore(config.db_path, embedder) as store:
        record = store.get(args.memory_id)
        content = record.content if record else ""
        store.delete(args.memory_id)
        print(f"forgot memory {args.memory_id}: {content}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import SESSION_1_TURNS, run_two_session_demo

    result = run_two_session_demo(db_path=args.db_path, offline=not args.live)

    for i, turn in enumerate(result.turns, start=1):
        print(f"=== turn {i}: {turn.session_id} ===")
        print(turn.stdout.rstrip())
        print()

    session2_reply_line = next(
        line for line in result.session2_reply.stdout.splitlines() if line.startswith("assistant: ")
    )
    print("=== summary ===")
    print(f"session 1 established facts across {len(SESSION_1_TURNS)} turns, then exited.")
    print("session 2 started with an empty conversation history (a fresh process)")
    print("and answered using only persistent memory retrieved from disk:")
    print(f"  {session2_reply_line}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory_agent")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="send one message and get a reply")
    p_chat.add_argument("message", help="the message to send")
    p_chat.add_argument("--session-id", default="default", help="label for this conversation")
    p_chat.add_argument("--offline", action="store_true", help="force the offline stub model")
    p_chat.set_defaults(func=_cmd_chat)

    p_memories = sub.add_parser("memories", help="list everything in persistent memory")
    p_memories.set_defaults(func=_cmd_memories)

    p_forget = sub.add_parser("forget", help="delete one persistent memory")
    p_forget.add_argument("memory_id")
    p_forget.set_defaults(func=_cmd_forget)

    p_demo = sub.add_parser("demo", help="run the two-session persistent-memory proof")
    p_demo.add_argument(
        "--db-path",
        default="memory_agent_demo.db",
        help="dedicated demo database (reset on every run, independent of MEMORY_AGENT_DB)",
    )
    p_demo.add_argument(
        "--offline",
        action="store_true",
        help="accepted for consistency with other commands -- the demo is offline by default",
    )
    p_demo.add_argument(
        "--live", action="store_true", help="use the real model if ANTHROPIC_API_KEY is set"
    )
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MemoryAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
