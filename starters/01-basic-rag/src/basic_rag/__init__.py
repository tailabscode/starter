"""basic_rag: a minimal, production-shaped hybrid RAG pipeline.

BM25 + hashed/Voyage dense embeddings fused with Reciprocal Rank Fusion,
grounded generation with resolvable [n] citations, and a fully offline
demo path via a deterministic stub LLM client.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
