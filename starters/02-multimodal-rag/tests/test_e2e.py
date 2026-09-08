"""End-to-end query path with the stub client: ingest -> query via the CLI,
fully offline, no network, over a corpus that mixes text and an image."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_png_bytes

from multimodal_rag import cli

MARKDOWN = (
    "# Inspection Notes\n\n"
    "## Chart summary\n\n"
    "The attached chart.png is a small revenue chart drawn mostly in blue "
    "bars for this quarter.\n"
)


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    images_dir = corpus / "images"
    images_dir.mkdir(parents=True)
    (corpus / "notes.md").write_text(MARKDOWN, encoding="utf-8")
    (images_dir / "chart.png").write_bytes(make_png_bytes(20, 10, (30, 90, 200)))
    return corpus


def test_ingest_then_query_offline_end_to_end(
    corpus_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    index_path = tmp_path / "index.json"

    ingest_code = cli.main(
        ["ingest", "--corpus", str(corpus_dir), "--index", str(index_path), "--offline"]
    )
    assert ingest_code == 0
    assert index_path.exists()
    ingest_out = capsys.readouterr().out
    assert "1 text chunks + 1 images" in ingest_out
    assert "hashing" in ingest_out

    query_code = cli.main(
        ["query", "What color is the revenue chart?", "--index", str(index_path), "--offline"]
    )
    assert query_code == 0
    query_out = capsys.readouterr().out

    assert "[stub]" in query_out
    assert "Retrieved:" in query_out
    assert "Citations:" in query_out
    # The image's caption is genuinely pixel-derived: this PNG is solid
    # rgb(30, 90, 200), so the stub's honest caption must say so.
    assert "rgb(30, 90, 200)" in query_out


def test_index_info_reports_text_and_image_counts(
    corpus_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index_path = tmp_path / "index.json"
    cli.main(["ingest", "--corpus", str(corpus_dir), "--index", str(index_path), "--offline"])
    capsys.readouterr()

    code = cli.main(["index-info", "--index", str(index_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert '"text_chunk_count": 1' in out
    assert '"image_count": 1' in out
    assert '"embedder": "hashing"' in out


def test_query_without_index_reports_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_index = tmp_path / "nope.json"
    code = cli.main(["query", "anything", "--index", str(missing_index), "--offline"])
    assert code == 1
    err = capsys.readouterr().err
    assert "ingest" in err.lower()
