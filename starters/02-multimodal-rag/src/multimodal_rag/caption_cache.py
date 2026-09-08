"""Disk cache for image captions, keyed by a content hash of the image bytes.

Vision calls are the expensive part of ingesting an image corpus. Keying
the cache on a hash of the actual bytes (not the filename) means editing
an image invalidates its cache entry automatically, and re-ingesting an
unchanged corpus never re-pays for vision. The cache key also includes
the provider ("stub" or "anthropic") so a StubClient caption is never
served back as if it came from real vision, or vice versa.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CACHE_DIR = Path(".cache/captions")


@dataclass(frozen=True)
class ImageCaption:
    caption: str
    extracted_text: str


def content_hash(image_bytes: bytes) -> str:
    """A short, stable id derived from the image's actual bytes."""
    return hashlib.sha256(image_bytes).hexdigest()[:24]


def _cache_path(cache_dir: Path, digest: str, provider: str) -> Path:
    return cache_dir / f"{digest}.{provider}.json"


def load_cached_caption(cache_dir: Path, digest: str, provider: str) -> ImageCaption | None:
    path = _cache_path(cache_dir, digest, provider)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ImageCaption(**data)


def save_cached_caption(cache_dir: Path, digest: str, provider: str, caption: ImageCaption) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, digest, provider)
    path.write_text(json.dumps(asdict(caption)), encoding="utf-8")
