"""The unified index must contain both text chunks and image items, and
survive a save/load round trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_png_bytes

from multimodal_rag.index import Index, ingest
from multimodal_rag.llm import StubClient


def _build_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    (corpus / "images").mkdir(parents=True)
    (corpus / "notes.md").write_text(
        "# Notes\n\n## Section\n\nThe quarterly chart is mostly blue this time.\n",
        encoding="utf-8",
    )
    (corpus / "images" / "chart.png").write_bytes(make_png_bytes(4, 4, (20, 80, 200)))
    return corpus


def test_ingest_produces_both_text_and_image_items(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    index = ingest(corpus_dir=corpus, offline=True, voyage_api_key=None, llm_client=StubClient())

    kinds = {item.kind for item in index.items}
    assert kinds == {"text", "image"}
    image_items = [item for item in index.items if item.kind == "image"]
    assert len(image_items) == 1
    assert image_items[0].source == "chart.png"
    assert "rgb(20, 80, 200)" in image_items[0].text  # the stub's real pixel-derived caption
    assert index.vectors.shape[0] == len(index.items)


def test_index_save_and_load_round_trips_both_kinds(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    index = ingest(corpus_dir=corpus, offline=True, voyage_api_key=None, llm_client=StubClient())
    index_path = tmp_path / "index.json"
    index.save(index_path)

    loaded = Index.load(index_path)
    assert {item.kind for item in loaded.items} == {"text", "image"}
    assert loaded.vectors.shape == index.vectors.shape
    loaded_image = next(item for item in loaded.items if item.kind == "image")
    assert loaded_image.image_path is not None
    assert Path(loaded_image.image_path).exists()


def test_ingest_raises_for_empty_corpus(tmp_path: Path) -> None:
    empty_corpus = tmp_path / "empty"
    empty_corpus.mkdir()
    with pytest.raises(ValueError, match="No .md/.txt files or images"):
        ingest(corpus_dir=empty_corpus, offline=True, voyage_api_key=None, llm_client=StubClient())
