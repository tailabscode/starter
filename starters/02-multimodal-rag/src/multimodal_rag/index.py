"""Build, persist and load the unified hybrid index.

Text chunks and image captions live in one `items` list and one dense
vector matrix -- indexed identically once an image has been captioned to
text. Persisted as a single JSON file, matching 01-basic-rag's format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .chunking import chunk_corpus
from .embeddings import Embedder, HashingEmbedder, VoyageEmbedder
from .errors import EmbedderMismatchError, MissingCredentialsError
from .image_ingest import ingest_images
from .items import Item

if TYPE_CHECKING:
    from .llm import LLMClient


@dataclass(frozen=True)
class IndexMetadata:
    embedder_name: str
    embedder_dim: int


class Index:
    """In-memory index: items + their dense vectors, loadable from/savable to JSON."""

    def __init__(self, items: list[Item], vectors: np.ndarray, metadata: IndexMetadata) -> None:
        self.items = items
        self.vectors = vectors
        self.metadata = metadata

    @property
    def item_ids(self) -> list[str]:
        return [item.item_id for item in self.items]

    @property
    def item_texts(self) -> list[str]:
        return [item.text for item in self.items]

    def get_item(self, item_id: str) -> Item | None:
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None

    def save(self, path: Path) -> None:
        data = {
            "metadata": asdict(self.metadata),
            "items": [asdict(item) for item in self.items],
            "vectors": self.vectors.tolist(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Index:
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = IndexMetadata(**data["metadata"])
        items = [Item(**item) for item in data["items"]]
        vectors = np.array(data["vectors"], dtype=np.float64)
        return cls(items=items, vectors=vectors, metadata=metadata)


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


def ingest(
    corpus_dir: Path,
    offline: bool,
    voyage_api_key: str | None,
    llm_client: LLMClient,
) -> Index:
    """Chunk text files, caption images, and embed everything into a fresh Index."""
    text_items = chunk_corpus(corpus_dir)
    image_items = ingest_images(corpus_dir / "images", llm_client)
    items = text_items + image_items
    if not items:
        raise ValueError(f"No .md/.txt files or images/*.png found in {corpus_dir}")

    kind = "hashing" if offline or not voyage_api_key else "voyage"
    embedder = build_embedder(kind, voyage_api_key)
    vectors = embedder.embed([item.text for item in items])
    metadata = IndexMetadata(embedder_name=embedder.name, embedder_dim=embedder.dim)
    return Index(items=items, vectors=vectors, metadata=metadata)


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
