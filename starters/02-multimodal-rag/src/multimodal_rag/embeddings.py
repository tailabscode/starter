"""Embedding backends.

Anthropic has no embeddings endpoint, so this starter offers two swappable
backends behind the same tiny protocol:

- HashingEmbedder (default): offline, deterministic, pure numpy. It hashes
  each token into one of `dim` buckets and L2-normalizes the resulting
  bag-of-words vector. It has NO semantic understanding, which is why
  retrieval is hybrid (see retrieval.py): BM25 carries lexical matches,
  this only adds a weak, offline-friendly secondary signal. It embeds
  image captions exactly like text -- captions are just text once they
  exist, which is the whole point of the caption-then-index pattern.
- VoyageEmbedder: real semantic embeddings via the Voyage AI HTTP API
  (voyage-3.5), called with stdlib urllib. Activated once VOYAGE_API_KEY
  is set.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .errors import MissingCredentialsError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Minimal interface both embedding backends satisfy."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class HashingEmbedder:
    """Deterministic, offline, hashed bag-of-words embedder. No semantic understanding."""

    dim: int = 256
    name: str = "hashing"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for token in _tokenize(text):
                bucket = int(hashlib.sha1(token.encode()).hexdigest(), 16) % self.dim
                vectors[i, bucket] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


@dataclass
class VoyageEmbedder:
    """Real semantic embeddings via Voyage AI. Requires VOYAGE_API_KEY."""

    api_key: str | None
    model: str = "voyage-3.5"
    dim: int = 1024
    name: str = "voyage"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.api_key:
            raise MissingCredentialsError(
                "VOYAGE_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://dash.voyageai.com/, or run with --offline to use the "
                "offline HashingEmbedder instead."
            )
        payload = json.dumps({"input": texts, "model": self.model}).encode()
        request = urllib.request.Request(
            "https://api.voyageai.com/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Voyage API error {exc.code}: {exc.read().decode()}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach the Voyage API: {exc.reason}") from exc

        vectors = np.array([item["embedding"] for item in body["data"]], dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
