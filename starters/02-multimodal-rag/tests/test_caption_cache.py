"""Caption cache hit/miss behavior, and that ingest_images actually uses it."""

from __future__ import annotations

from pathlib import Path

from conftest import make_png_bytes

from multimodal_rag.caption_cache import (
    ImageCaption,
    content_hash,
    load_cached_caption,
    save_cached_caption,
)
from multimodal_rag.image_ingest import ingest_images


def test_cache_miss_then_hit_round_trips(tmp_path: Path) -> None:
    digest = content_hash(b"some image bytes")
    assert load_cached_caption(tmp_path, digest, "stub") is None

    caption = ImageCaption(caption="a blue chart", extracted_text="")
    save_cached_caption(tmp_path, digest, "stub", caption)

    loaded = load_cached_caption(tmp_path, digest, "stub")
    assert loaded == caption


def test_cache_is_scoped_per_provider(tmp_path: Path) -> None:
    digest = content_hash(b"some image bytes")
    save_cached_caption(tmp_path, digest, "stub", ImageCaption("stub caption", ""))
    # A different provider must not see the stub's cached entry.
    assert load_cached_caption(tmp_path, digest, "anthropic") is None


class _CountingClient:
    provider = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def caption_image(self, image_bytes: bytes, media_type: str) -> ImageCaption:
        self.calls += 1
        return ImageCaption(caption=f"call number {self.calls}", extracted_text="")


def test_ingest_images_calls_captioner_once_then_reuses_cache(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.png").write_bytes(make_png_bytes(2, 2, (255, 0, 0)))
    cache_dir = tmp_path / "cache"
    client = _CountingClient()

    first = ingest_images(images_dir, client, cache_dir=cache_dir)
    assert client.calls == 1
    assert first[0].text == "call number 1"

    second = ingest_images(images_dir, client, cache_dir=cache_dir)
    assert client.calls == 1  # cache hit, no second call
    assert second[0].text == "call number 1"


def test_changing_image_bytes_invalidates_the_cache(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    path = images_dir / "a.png"
    cache_dir = tmp_path / "cache"
    client = _CountingClient()

    path.write_bytes(make_png_bytes(2, 2, (255, 0, 0)))
    ingest_images(images_dir, client, cache_dir=cache_dir)
    assert client.calls == 1

    path.write_bytes(make_png_bytes(2, 2, (0, 255, 0)))  # different content -> different hash
    ingest_images(images_dir, client, cache_dir=cache_dir)
    assert client.calls == 2


def test_ingest_images_returns_empty_list_for_missing_dir(tmp_path: Path) -> None:
    client = _CountingClient()
    result = ingest_images(tmp_path / "does-not-exist", client, cache_dir=tmp_path / "cache")
    assert result == []
    assert client.calls == 0
