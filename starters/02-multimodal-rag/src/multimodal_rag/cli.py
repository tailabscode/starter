"""Command-line interface: ingest, query, index-info.

Also runnable as `python -m multimodal_rag <subcommand>`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config
from .errors import EmbedderMismatchError, MissingCredentialsError
from .generation import build_content_items, make_answer_result
from .index import Index, embedder_for_query
from .index import ingest as ingest_corpus
from .llm import get_client
from .logging_setup import configure_logging
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = Path("data/corpus")
DEFAULT_INDEX = Path("data/corpus.index.json")


def _cmd_ingest(args: argparse.Namespace, config: Config) -> int:
    llm_client = get_client(config, offline=args.offline)
    index = ingest_corpus(
        corpus_dir=Path(args.corpus),
        offline=args.offline,
        voyage_api_key=config.voyage_api_key,
        llm_client=llm_client,
    )
    index_path = Path(args.index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)

    text_count = sum(1 for item in index.items if item.kind == "text")
    image_count = sum(1 for item in index.items if item.kind == "image")
    print(
        f"Indexed {text_count} text chunks + {image_count} images from {args.corpus} "
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

    items = [item for r in retrieved if (item := index.get_item(r.item_id)) is not None]
    content_items = build_content_items(items)

    client = get_client(config, offline=args.offline)
    answer_text = client.generate_answer(args.query, content_items)
    result = make_answer_result(answer_text, content_items)

    print(result.text)
    print()
    print("Retrieved:")
    for ci in content_items:
        print(f"  [{ci.number}] ({ci.item.kind}) {ci.item.source}")
    print()
    print("Citations:")
    for citation in result.citations:
        if citation.resolved:
            print(f"  [{citation.marker}] -> {citation.source} (item {citation.item_id})")
        else:
            print(f"  [{citation.marker}] -> UNRESOLVED (cited an item that wasn't retrieved)")
    if result.unresolved_markers:
        print(f"Warning: unresolved citation markers: {result.unresolved_markers}", file=sys.stderr)
    return 0


def _cmd_index_info(args: argparse.Namespace, config: Config) -> int:
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"No index found at {index_path}. Run `ingest` first.", file=sys.stderr)
        return 1

    index = Index.load(index_path)
    text_items = [item for item in index.items if item.kind == "text"]
    image_items = [item for item in index.items if item.kind == "image"]
    info = {
        "index_path": str(index_path),
        "item_count": len(index.items),
        "text_chunk_count": len(text_items),
        "image_count": len(image_items),
        "text_sources": sorted({item.source for item in text_items}),
        "image_sources": sorted({item.source for item in image_items}),
        "embedder": index.metadata.embedder_name,
        "embedder_dim": index.metadata.embedder_dim,
    }
    print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimodal-rag", description="Hybrid RAG over text and images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Chunk text, caption images, and index a corpus."
    )
    ingest_parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Corpus directory.")
    ingest_parser.add_argument(
        "--index", default=str(DEFAULT_INDEX), help="Where to write the index JSON."
    )
    ingest_parser.add_argument(
        "--offline",
        action="store_true",
        help="Force the offline stub captioner and the hashing embedder (no network).",
    )
    ingest_parser.set_defaults(func=_cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Answer a question from a saved index.")
    query_parser.add_argument("query", help="Question to answer.")
    query_parser.add_argument("--index", default=str(DEFAULT_INDEX), help="Index JSON to read.")
    query_parser.add_argument(
        "--top-k", type=int, default=None, help="Items to retrieve (default from TOP_K env, 5)."
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
