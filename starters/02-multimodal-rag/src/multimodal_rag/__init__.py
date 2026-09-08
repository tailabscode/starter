"""multimodal_rag: hybrid RAG over a corpus that mixes text and images.

Images are captioned by Claude vision into searchable text at ingest time
and indexed alongside text chunks in one hybrid (BM25 + dense) index; at
query time, retrieved image items get their real pixels re-attached to
the generation call. A fully offline demo path (--offline) uses a stub
vision client that decodes real PNG bytes for deterministic captions.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
