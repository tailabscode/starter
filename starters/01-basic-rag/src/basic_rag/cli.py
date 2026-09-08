"""Command-line interface: ingest, query, index-info.

Also runnable as `python -m basic_rag <subcommand>`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config
from .errors import EmbedderMismatchError, MissingCredentialsError
from .generation import build_context_blocks, make_answer_result
from .index import Index, embedder_for_query
from .index import ingest as ingest_corpus
from .llm import get_client
from .logging_setup import configure_logging
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = Path("data/corpus")
DEFAULT_INDEX = Path("data/corpus.index.json")


def _cmd_ingest(args: argparse.Namespace, config: Config) -> int:
    index = ingest_corpus(
        corpus_dir=Path(args.corpus),
        offline=args.offline,
        voyage_api_key=config.voyage_api_key,
    )
    index_path = Path(args.index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)
    print(
        f"Indexed {len(index.chunks)} chunks from {args.corpus} "
        f"using the '{index.metadata.embedder_name}' embedder -> {args.index}"
    )
    return 0


def _cmd_query(args: argparse.Namespace, config: Config) -> int:
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"No index found at {index_path}. Run `ingest` first.", file=sys.stderr)
        return 1

    index = Index.load(index_path)
    embedder = embedder_for_query(index, offline=args.offline, voyage_api_key=config.voyage_api_key)
    retriever = HybridRetriever(index=index, embedder=embedder)
    retrieved = retriever.retrieve(args.query, top_k=args.top_k, rrf_k=args.rrf_k)

    chunks = [chunk for r in retrieved if (chunk := index.get_chunk(r.chunk_id)) is not None]
    blocks = build_context_blocks(chunks)

    client = get_client(config, offline=args.offline)
    answer_text = client.generate_answer(args.query, blocks)
    result = make_answer_result(answer_text, blocks)

    print(result.text)
    print()
    print("Citations:")
    for citation in result.citations:
        if citation.resolved:
            print(f"  [{citation.marker}] -> {citation.source} (chunk {citation.chunk_id})")
        else:
            print(f"  [{citation.marker}] -> UNRESOLVED (cited a block that wasn't retrieved)")
    if result.unresolved_markers:
        print(f"Warning: unresolved citation markers: {result.unresolved_markers}", file=sys.stderr)
    return 0


def _cmd_index_info(args: argparse.Namespace, config: Config) -> int:
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"No index found at {index_path}. Run `ingest` first.", file=sys.stderr)
        return 1

    index = Index.load(index_path)
    sources = sorted({c.source for c in index.chunks})
    info = {
        "index_path": str(index_path),
        "chunk_count": len(index.chunks),
        "source_count": len(sources),
        "sources": sources,
        "embedder": index.metadata.embedder_name,
        "embedder_dim": index.metadata.embedder_dim,
    }
    print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="basic-rag", description="Minimal hybrid RAG pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Chunk, embed and index a corpus.")
    ingest_parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Corpus directory.")
    ingest_parser.add_argument(
        "--index", default=str(DEFAULT_INDEX), help="Where to write the index JSON."
    )
    ingest_parser.add_argument(
        "--offline", action="store_true", help="Force the offline HashingEmbedder (no network)."
    )
    ingest_parser.set_defaults(func=_cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Answer a question from a saved index.")
    query_parser.add_argument("query", help="Question to answer.")
    query_parser.add_argument("--index", default=str(DEFAULT_INDEX), help="Index JSON to read.")
    query_parser.add_argument(
        "--top-k", type=int, default=None, help="Chunks to retrieve (default from TOP_K env, 5)."
    )
    query_parser.add_argument(
        "--rrf-k", type=int, default=None, help="RRF k constant (default from RRF_K env, 60)."
    )
    query_parser.add_argument(
        "--offline",
        action="store_true",
        help="Force the offline stub LLM and hashing embedder (no network).",
    )
    query_parser.set_defaults(func=_cmd_query)

    info_parser = subparsers.add_parser("index-info", help="Show stats about a saved index.")
    info_parser.add_argument("--index", default=str(DEFAULT_INDEX), help="Index JSON to read.")
    info_parser.add_argument(
        "--offline",
        action="store_true",
        help="Accepted for CLI consistency; this command makes no network calls either way.",
    )
    info_parser.set_defaults(func=_cmd_index_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Config.from_env()
    configure_logging(config.log_level)

    if hasattr(args, "top_k") and args.top_k is None:
        args.top_k = config.top_k
    if hasattr(args, "rrf_k") and args.rrf_k is None:
        args.rrf_k = config.rrf_k

    try:
        return args.func(args, config)
    except (MissingCredentialsError, EmbedderMismatchError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
