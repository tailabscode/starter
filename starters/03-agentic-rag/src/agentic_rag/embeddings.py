"""Two embedding backends, matching the collection-wide retrieval convention.

Anthropic does not serve an embeddings endpoint, so a RAG starter has to be honest about what
its default embedder actually does:

* :class:`HashingEmbedder` is offline, deterministic, and has **no semantic understanding at
  all** -- it is a hashed bag-of-words. Two sentences that share no words score zero similarity
  even if they mean the same thing. It exists purely so the starter runs with no API key.
* :class:`VoyageEmbedder` calls Voyage AI's real embedding model over plain ``urllib.request``
  and is the documented upgrade path once a ``VOYAGE_API_KEY`` is available.

Because the default embedder is weak, ``retrieval.py`` never relies on it alone -- it is fused
with BM25 via reciprocal rank fusion, and BM25 does most of the real work offline.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Protocol

import numpy as np

from agentic_rag.bm25 import tokenize


class Embedder(Protocol):
    """Anything that can turn a batch of texts into unit-normalized vectors."""

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic hashed bag-of-words, projected to a fixed dimension, L2-normalized.

    Uses ``hashlib.blake2b`` rather than Python's built-in ``hash()`` because the latter is
    randomized per-process (``PYTHONHASHSEED``) and would make retrieval non-reproducible.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _hash_token(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                index, sign = self._hash_token(token)
                vectors[row, index] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class VoyageEmbedder:
    """Real semantic embeddings via Voyage AI's ``voyage-3.5`` model.

    This is the upgrade path once semantics matter: activate it by setting ``VOYAGE_API_KEY``.
    Uses stdlib ``urllib.request`` only, per the dependency budget (no ``requests``).
    """

    _ENDPOINT = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = "voyage-3.5") -> None:
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        body = json.dumps({"input": texts, "model": self.model}).encode("utf-8")
        request = urllib.request.Request(
            self._ENDPOINT,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        vectors = np.array([item["embedding"] for item in payload["data"]], dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
