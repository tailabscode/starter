"""End-to-end query path with the stub client: ingest -> query via the CLI,
fully offline, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from basic_rag import cli

CORPUS = {
    "policy.md": (
        "# Travel Policy\n\n"
        "## Booking\n\n"
        "All flights must be booked through the travel portal at least two "
        "weeks in advance whenever possible.\n\n"
        "## Reimbursement\n\n"
        "Submit receipts within 30 days of travel for reimbursement.\n"
    ),
    "handbook.md": (
        "# Employee Handbook\n\n"
        "## Working hours\n\n"
        "Core collaboration hours are 10:00 to 13:00 UTC for all "
        "employees regardless of timezone.\n"
    ),
}


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in CORPUS.items():
        (corpus / name).write_text(text, encoding="utf-8")
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
    assert "hashing" in ingest_out

    query_code = cli.main(
        ["query", "When should I book flights?", "--index", str(index_path), "--offline"]
    )
    assert query_code == 0
    query_out = capsys.readouterr().out

    assert "[stub]" in query_out
    assert "Citations:" in query_out
    # The stub echoes retrieved chunk content, so the travel-booking chunk
    # should show up in the answer for this query.
    assert "travel portal" in query_out


def test_index_info_reports_chunk_and_source_counts(
    corpus_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index_path = tmp_path / "index.json"
    cli.main(["ingest", "--corpus", str(corpus_dir), "--index", str(index_path), "--offline"])
    capsys.readouterr()

    code = cli.main(["index-info", "--index", str(index_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert '"source_count": 2' in out
    assert '"embedder": "hashing"' in out


def test_query_without_index_reports_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_index = tmp_path / "nope.json"
    code = cli.main(["query", "anything", "--index", str(missing_index), "--offline"])
    assert code == 1
    err = capsys.readouterr().err
    assert "ingest" in err.lower()
