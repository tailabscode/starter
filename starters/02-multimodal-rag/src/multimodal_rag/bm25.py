"""A small Okapi BM25 implementation. No dependencies beyond the stdlib.

See 01-basic-rag/src/basic_rag/bm25.py for the full rationale (duplicated
here on purpose -- each starter in this library is standalone). BM25 does
the real lexical retrieval work over both text-chunk and image-caption
text; dense similarity is fused on top of it (see retrieval.py).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric-run tokenizer shared by BM25 and query encoding."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 ranking over a fixed corpus of pre-tokenized documents."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_term_counts: list[Counter[str]] = [Counter(doc) for doc in documents]
        self.doc_lengths: list[int] = [len(doc) for doc in documents]
        self.n_docs = len(documents)
        self.avg_doc_length = (sum(self.doc_lengths) / self.n_docs) if self.n_docs else 0.0

        doc_freq: Counter[str] = Counter()
        for doc in documents:
            doc_freq.update(set(doc))
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score_all(self, query_tokens: list[str]) -> list[float]:
        """Return a BM25 score for every document, in corpus order."""
        scores = [0.0] * self.n_docs
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.doc_term_counts):
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[i] / (self.avg_doc_length or 1)
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores

    def rank(self, query_tokens: list[str]) -> list[int]:
        """Return document indices sorted best-to-worst by BM25 score."""
        scores = self.score_all(query_tokens)
        return sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)
