"""Hashing bag-of-words embedder + cosine similarity, in pure Python.

Same idea as the `HashingEmbedder` in the RAG starters (01/02/03) -- hash each
token into a fixed-size vector, L2-normalize -- but reimplemented standalone
here with plain lists and `math`/`hashlib`, no numpy: this starter isn't one
of the vector-maths starters the shared build spec allows numpy for, and at
this starter's scale (a handful to a few hundred memory records) pure Python
is plenty fast.

It has **no semantic understanding** -- it cannot tell "car" and "automobile"
are related. It only catches literal token overlap. That is enough to rank a
memory that shares real words with a query above one that shares none, which
is all this starter's retrieval needs; see the README's Limitations section.

`hashlib.sha256` is used instead of Python's builtin `hash()` on purpose:
builtin string hashing is randomized per-process (`PYTHONHASHSEED`), which
would make embeddings of the same text differ across runs -- fatal for a
starter whose demo spans multiple processes sharing one SQLite file.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic, offline, dependency-free text embedder."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors. Equals cosine similarity when both are unit norm."""
    return sum(x * y for x, y in zip(a, b, strict=True))
