"""Turn a directory of images into indexable Items via vision captioning.

Each image is captioned once (StubClient or AnthropicClient, see llm.py)
into a searchable text caption + any extracted visible text; the caption
is what gets indexed alongside text chunks (see items.py / index.py). A
disk cache keyed by content hash (caption_cache.py) means re-ingesting an
unchanged corpus never re-pays for vision.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .caption_cache import DEFAULT_CACHE_DIR, content_hash, load_cached_caption, save_cached_caption
from .images import guess_media_type
from .items import Item

if TYPE_CHECKING:
    from .llm import LLMClient


def find_images(images_dir: Path) -> list[Path]:
    """List .png files directly inside images_dir, sorted for determinism."""
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")


def ingest_images(
    images_dir: Path, client: LLMClient, cache_dir: Path = DEFAULT_CACHE_DIR
) -> list[Item]:
    """Caption every image in images_dir (using the cache where possible) as Items."""
    items: list[Item] = []
    for path in find_images(images_dir):
        image_bytes = path.read_bytes()
        media_type = guess_media_type(path)
        digest = content_hash(image_bytes)

        cached = load_cached_caption(cache_dir, digest, client.provider)
        if cached is not None:
            caption = cached
        else:
            caption = client.caption_image(image_bytes, media_type)
            save_cached_caption(cache_dir, digest, client.provider, caption)

        text = f"{caption.caption} {caption.extracted_text}".strip()
        items.append(
            Item(
                item_id=digest[:12],
                kind="image",
                source=path.name,
                text=text,
                heading_trail=[],
                image_path=str(path.resolve()),
                media_type=media_type,
            )
        )
    return items
