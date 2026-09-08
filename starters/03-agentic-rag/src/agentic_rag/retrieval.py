"""Corpus loading, chunking, and hybrid (BM25 + dense) retrieval with RRF fusion.

No vector database: the "vector store" is a numpy matrix held in memory, with JSON
persistence available via :meth:`HybridRetriever.save` / :meth:`HybridRetriever.load` for
callers that want to avoid recomputing embeddings across runs. The demo corpus is tiny enough
that the CLI simply rebuilds the index on every invocation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from agentic_rag.bm25 import BM25Index
from agentic_rag.embeddings import Embedder, HashingEmbedder
from agentic_rag.errors import CorpusNotFoundError, UnknownChunkError

_HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_RRF_K = 60


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit: a section of a markdown document."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    heading: str
    text: str
    position: int

    def to_dict(self) -> dict:
        return asdict(self)


def _split_sections(doc_text: str) -> list[tuple[str, str]]:
    """Split a markdown document into (heading, body) pairs on ``##`` boundaries."""
    matches = list(_HEADING_RE.finditer(doc_text))
    if not matches:
        return [("", doc_text.strip())]
    sections = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc_text)
        sections.append((match.group(1).strip(), doc_text[start:end].strip()))
    return sections


def load_corpus(data_dir: Path) -> list[Chunk]:
    """Load every ``*.md`` file in ``data_dir`` and split it into heading-bounded chunks."""
    paths = sorted(data_dir.glob("*.md"))
    if not paths:
        raise CorpusNotFoundError(f"No markdown documents found in {data_dir}")
    chunks: list[Chunk] = []
    for path in paths:
        doc_id = path.stem
        raw = path.read_text(encoding="utf-8")
        title_match = re.match(r"^#\s+(.*)$", raw, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else doc_id
        for position, (heading, body) in enumerate(_split_sections(raw)):
            if not body:
                continue
            chunk_text = f"{heading}\n{body}" if heading else body
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{position}",
                    doc_id=doc_id,
                    source=path.name,
                    title=title,
                    heading=heading or title,
                    text=chunk_text,
                    position=position,
                )
            )
    return chunks


class HybridRetriever:
    """BM25 + hashed-dense retrieval fused with Reciprocal Rank Fusion (RRF)."""

    def __init__(self, chunks: list[Chunk], *, embedder: Embedder | None = None) -> None:
        if not chunks:
            raise CorpusNotFoundError("Cannot build a retriever over zero chunks")
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._embedder = embedder or HashingEmbedder()
        texts = [c.text for c in chunks]
        self._bm25 = BM25Index(texts)
        self._dense = self._embedder.embed(texts)

    @classmethod
    def from_data_dir(cls, data_dir: Path, *, embedder: Embedder | None = None) -> HybridRetriever:
        return cls(load_corpus(data_dir), embedder=embedder)

    def _dense_scores(self, query: str) -> np.ndarray:
        query_vec = self._embedder.embed([query])[0]
        return self._dense @ query_vec

    @staticmethod
    def _ranks_from_scores(scores: list[float]) -> dict[int, int]:
        """Map document index -> 1-based rank, descending by score, ties broken by index."""
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return {doc_index: rank + 1 for rank, doc_index in enumerate(order)}

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid search: fuse BM25 and dense cosine rankings with RRF, return top_k hits."""
        bm25_scores = self._bm25.score(query)
        dense_scores = self._dense_scores(query).tolist()
        bm25_ranks = self._ranks_from_scores(bm25_scores)
        dense_ranks = self._ranks_from_scores(dense_scores)

        fused: list[tuple[int, float]] = []
        for i in range(len(self.chunks)):
            rrf_score = 1.0 / (_RRF_K + bm25_ranks[i]) + 1.0 / (_RRF_K + dense_ranks[i])
            fused.append((i, rrf_score))
        fused.sort(key=lambda pair: pair[1], reverse=True)

        results = []
        for i, score in fused[:top_k]:
            chunk = self.chunks[i]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "heading": chunk.heading,
                    "text": chunk.text,
                    "score": round(score, 6),
                    "bm25_score": round(bm25_scores[i], 4),
                }
            )
        return results

    def list_topics(self) -> list[dict]:
        """Summarize what the corpus covers: one entry per source document."""
        seen: dict[str, dict] = {}
        for chunk in self.chunks:
            if chunk.doc_id not in seen:
                seen[chunk.doc_id] = {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "headings": [],
                }
            seen[chunk.doc_id]["headings"].append(chunk.heading)
        return list(seen.values())

    def fetch_chunk(self, chunk_id: str, context: int = 1) -> dict:
        """Return a chunk plus up to ``context`` neighbouring chunks from the same document."""
        chunk = self._by_id.get(chunk_id)
        if chunk is None:
            raise UnknownChunkError(chunk_id)
        neighbours = [
            c.to_dict()
            for c in self.chunks
            if c.doc_id == chunk.doc_id and abs(c.position - chunk.position) <= context
        ]
        neighbours.sort(key=lambda d: d["position"])
        return {"chunk_id": chunk_id, "chunk": chunk.to_dict(), "context": neighbours}

    def valid_chunk_ids(self) -> set[str]:
        return set(self._by_id)

    def save(self, path: Path) -> None:
        """Persist chunk metadata and dense vectors to a single JSON file."""
        payload = {
            "chunks": [c.to_dict() for c in self.chunks],
            "dense_vectors": self._dense.tolist(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, embedder: Embedder | None = None) -> HybridRetriever:
        """Rebuild a retriever from a JSON file written by :meth:`save`.

        Dense vectors are restored directly (no re-embedding); BM25 is rebuilt from the chunk
        text since it is cheap and has no persisted state of its own.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in payload["chunks"]]
        instance = cls(chunks, embedder=embedder)
        instance._dense = np.array(payload["dense_vectors"], dtype=np.float64)
        return instance
