"""A small local corpus and a deliberately simple search over it.

This starter is not about retrieval -- it's about coordination, concurrency, and critique --
so the evidence tool is a plain term-overlap scorer, not the hybrid BM25 + dense pipeline from
starter 03 (which this repository intentionally does not import; see that starter's README for
the fuller retrieval approach). Each researcher gets this as its one tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from multi_agent_research.errors import CorpusNotFoundError

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "which",
}


def tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    title: str
    text: str


class Corpus:
    """Loads every ``*.md`` file in a directory and scores them by stopword-filtered overlap."""

    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise CorpusNotFoundError("Cannot build a corpus over zero documents")
        self.documents = documents

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> Corpus:
        paths = sorted(data_dir.glob("*.md"))
        if not paths:
            raise CorpusNotFoundError(f"No markdown documents found in {data_dir}")
        docs = []
        for path in paths:
            raw = path.read_text(encoding="utf-8")
            title_match = re.match(r"^#\s+(.*)$", raw, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else path.stem
            docs.append(Document(doc_id=path.stem, source=path.name, title=title, text=raw))
        return cls(docs)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Score every document by the fraction of query terms it contains, descending."""
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scored = []
        for doc in self.documents:
            doc_terms = tokenize(doc.text)
            overlap = query_terms & doc_terms
            if not overlap:
                continue
            score = len(overlap) / len(query_terms)
            scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "title": doc.title,
                "text": doc.text.strip(),
                "score": round(score, 4),
            }
            for score, doc in scored[:top_k]
        ]

    def topics(self) -> list[dict[str, str]]:
        return [{"doc_id": doc.doc_id, "title": doc.title} for doc in self.documents]
