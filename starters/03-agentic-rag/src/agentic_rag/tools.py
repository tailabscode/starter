"""Tool definitions exposed to Claude, and the dispatcher that executes them.

Every tool is strict (``"strict": True`` with ``additionalProperties: False``) so tool inputs
are guaranteed to validate against the schema before we ever see them.
"""

from __future__ import annotations

from agentic_rag.errors import UnknownChunkError
from agentic_rag.retrieval import HybridRetriever

TOOL_DEFS: list[dict] = [
    {
        "name": "search_corpus",
        "description": (
            "Hybrid (BM25 + dense) search over the corpus. Returns the top matching chunks "
            "with their chunk id, source filename, and text. Use this to find facts."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (1-10).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_topics",
        "description": (
            "List every document in the corpus and its section headings, with no search "
            "query. Use this to check whether a topic is covered by the corpus at all before "
            "concluding a search returned 'no results' by mistake, or to decide the question "
            "is simply out of scope for this corpus."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_chunk",
        "description": (
            "Fetch a specific chunk by id, plus its neighbouring chunks from the same "
            "document, for extra context around a promising search hit."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "A chunk id from search_corpus."},
            },
            "required": ["chunk_id"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_DEFS]


def execute_tool(name: str, tool_input: dict, retriever: HybridRetriever) -> tuple[dict, bool]:
    """Run a tool call against the retriever. Returns (result, is_error)."""
    try:
        if name == "search_corpus":
            query = tool_input["query"]
            top_k = int(tool_input.get("top_k", 5))
            return {"results": retriever.search(query, top_k=top_k)}, False
        if name == "list_topics":
            return {"topics": retriever.list_topics()}, False
        if name == "fetch_chunk":
            return retriever.fetch_chunk(tool_input["chunk_id"]), False
        return {"error": f"Unknown tool: {name}"}, True
    except UnknownChunkError as exc:
        return {"error": f"No such chunk id: {exc}"}, True
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": f"Invalid tool input: {exc}"}, True
