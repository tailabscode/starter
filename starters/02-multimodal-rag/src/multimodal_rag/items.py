"""The unified retrievable unit: either a text chunk or an image caption.

Keeping both kinds in one dataclass, one list, and one index is the point
of this starter's ingest pattern: text chunks and image captions are
indexed exactly the same way (BM25 + dense over `text`), so retrieval
doesn't need to know which modality it found until generation time, when
image items get their real pixels re-attached (see generation.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    item_id: str
    kind: str  # "text" | "image"
    source: str
    text: str  # what gets indexed: chunk text, or "caption. extracted_text"
    heading_trail: list[str]
    image_path: str | None = None
    media_type: str | None = None
