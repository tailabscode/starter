"""Build, persist and load the hybrid index: chunks plus their dense vectors.

BM25 needs no persistence -- it's rebuilt from the persisted chunk texts in
a few milliseconds by HybridRetriever (see retrieval.py) -- so the on-disk
format only needs to carry chunks and dense vectors, kept as a single JSON
file per the spec (no vector database).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_corpus
from .embeddings import Embedder, HashingEmbedder, VoyageEmbedder
from .errors import EmbedderMismatchError, MissingCredentialsError


@dataclass(frozen=True)
class IndexMetadata:
    embedder_name: str
    embedder_dim: int


class Index:
    """In-memory index: chunks + their dense vectors, loadable from/savable to JSON."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, metadata: IndexMetadata) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.metadata = metadata

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def chunk_texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def save(self, path: Path) -> None:
        data = {
            "metadata": asdict(self.metadata),
            "chunks": [asdict(c) for c in self.chunks],
            "vectors": self.vectors.tolist(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Index:
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = IndexMetadata(**data["metadata"])
        chunks = [Chunk(**c) for c in data["chunks"]]
        vectors = np.array(data["vectors"], dtype=np.float64)
        return cls(chunks=chunks, vectors=vectors, metadata=metadata)


def build_embedder(kind: str, voyage_api_key: str | None) -> Embedder:
    """Instantiate the embedder named by `kind` ("hashing" or "voyage")."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "voyage":
        if not voyage_api_key:
            raise MissingCredentialsError(
                "VOYAGE_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or use --offline to build/query a hashing-based index instead."
            )
        return VoyageEmbedder(api_key=voyage_api_key)
    raise ValueError(f"Unknown embedder kind: {kind!r}")


def ingest(corpus_dir: Path, offline: bool, voyage_api_key: str | None) -> Index:
    """Load, chunk and embed a corpus directory into a fresh Index."""
    chunks = chunk_corpus(corpus_dir)
    if not chunks:
        raise ValueError(f"No .md/.txt files found in {corpus_dir}")

    kind = "hashing" if offline or not voyage_api_key else "voyage"
    embedder = build_embedder(kind, voyage_api_key)
    vectors = embedder.embed([c.text for c in chunks])
    metadata = IndexMetadata(embedder_name=embedder.name, embedder_dim=embedder.dim)
    return Index(chunks=chunks, vectors=vectors, metadata=metadata)


def embedder_for_query(index: Index, offline: bool, voyage_api_key: str | None) -> Embedder:
    """Pick the embedder to encode a query with, matching how the index was built."""
    kind = index.metadata.embedder_name
    if offline and kind != "hashing":
        raise EmbedderMismatchError(
            f"Index was built with the '{kind}' embedder, which needs network access. "
            "Re-ingest with --offline to build a hashing-based index for offline queries."
        )
    embedder = build_embedder(kind, voyage_api_key)
    if embedder.dim != index.metadata.embedder_dim:
        raise ValueError(
            f"Embedder dim {embedder.dim} does not match index dim "
            f"{index.metadata.embedder_dim}; re-ingest the corpus."
        )
    return embedder
