from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from reportlab.pdfgen import canvas

from app.config import Settings
from app.errors import KnowledgeBackendUnavailableError
from app.ingestion import ingest_faq_directory
from app.pdf_extraction import extract_pdf


def make_pdf(path: Path, text: str) -> None:
    c = canvas.Canvas(str(path), pagesize=(300, 300))
    c.drawString(30, 250, text)
    c.showPage()
    c.save()


def make_settings(faq_dir: Path) -> Settings:
    return Settings(faq_pdf_dir=str(faq_dir), chunk_size=1000, chunk_overlap=150, openai_api_key="sk-test")


@pytest.fixture
def faq_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "faq_pdfs"
    directory.mkdir()
    return directory


async def test_new_file_gets_indexed(faq_dir: Path):
    make_pdf(faq_dir / "faq.pdf", "Como renegociar minhas parcelas em atraso?")
    settings = make_settings(faq_dir)

    with (
        patch("app.ingestion.get_indexed_hash", AsyncMock(return_value=None)) as mock_get_hash,
        patch("app.ingestion.embed_texts", AsyncMock(return_value=[[0.1, 0.2]])) as mock_embed,
        patch("app.ingestion.delete_chunks_for_file", AsyncMock()) as mock_delete,
        patch("app.ingestion.index_chunk", AsyncMock()) as mock_index,
        patch("app.ingestion.refresh_index", AsyncMock()) as mock_refresh,
    ):
        summary = await ingest_faq_directory(openai_client=object(), opensearch_client=object(), settings=settings)

    assert summary.files_indexed == 1
    assert summary.files_skipped == 0
    assert summary.files_failed == 0
    assert summary.chunks_written == 1
    mock_get_hash.assert_awaited_once()
    mock_embed.assert_awaited_once()
    mock_delete.assert_not_awaited()  # nothing to delete for a brand-new file
    mock_index.assert_awaited_once()
    mock_refresh.assert_awaited_once()


async def test_unchanged_file_is_skipped(faq_dir: Path):
    pdf_path = faq_dir / "faq.pdf"
    make_pdf(pdf_path, "Conteudo estavel de FAQ.")
    existing_hash = extract_pdf(pdf_path).content_hash
    settings = make_settings(faq_dir)

    with (
        patch("app.ingestion.get_indexed_hash", AsyncMock(return_value=existing_hash)),
        patch("app.ingestion.count_indexed_chunks", AsyncMock(return_value=1)),  # matches len(chunks)
        patch("app.ingestion.embed_texts", AsyncMock()) as mock_embed,
    ):
        summary = await ingest_faq_directory(openai_client=object(), opensearch_client=object(), settings=settings)

    assert summary.files_skipped == 1
    assert summary.files_indexed == 0
    mock_embed.assert_not_awaited()


async def test_hash_match_but_incomplete_chunk_count_is_reingested(faq_dir: Path):
    # Guards against a real bug found in manual verification: a chunk write can time out
    # client-side while still landing server-side, leaving fewer docs indexed than the
    # file's chunk count even though the hash "matches". Count must be checked too.
    pdf_path = faq_dir / "faq.pdf"
    make_pdf(pdf_path, "Conteudo estavel de FAQ.")
    existing_hash = extract_pdf(pdf_path).content_hash
    settings = make_settings(faq_dir)

    with (
        patch("app.ingestion.get_indexed_hash", AsyncMock(return_value=existing_hash)),
        patch("app.ingestion.count_indexed_chunks", AsyncMock(return_value=0)),  # partial: 0 of 1 chunk
        patch("app.ingestion.embed_texts", AsyncMock(return_value=[[0.1, 0.2]])) as mock_embed,
        patch("app.ingestion.delete_chunks_for_file", AsyncMock()) as mock_delete,
        patch("app.ingestion.index_chunk", AsyncMock()),
        patch("app.ingestion.refresh_index", AsyncMock()),
    ):
        summary = await ingest_faq_directory(openai_client=object(), opensearch_client=object(), settings=settings)

    assert summary.files_indexed == 1
    assert summary.files_skipped == 0
    mock_embed.assert_awaited_once()
    mock_delete.assert_awaited_once()  # stale partial chunk cleared before reindexing


async def test_changed_file_deletes_old_chunks_and_reindexes(faq_dir: Path):
    make_pdf(faq_dir / "faq.pdf", "Conteudo novo apos edicao do FAQ.")
    settings = make_settings(faq_dir)

    with (
        patch("app.ingestion.get_indexed_hash", AsyncMock(return_value="stale-hash-from-before")),
        patch("app.ingestion.embed_texts", AsyncMock(return_value=[[0.1, 0.2]])),
        patch("app.ingestion.delete_chunks_for_file", AsyncMock()) as mock_delete,
        patch("app.ingestion.index_chunk", AsyncMock()),
        patch("app.ingestion.refresh_index", AsyncMock()),
    ):
        summary = await ingest_faq_directory(openai_client=object(), opensearch_client=object(), settings=settings)

    assert summary.files_indexed == 1
    mock_delete.assert_awaited_once()


async def test_malformed_pdf_is_skipped_without_failing_the_run(faq_dir: Path):
    (faq_dir / "corrupt.pdf").write_bytes(b"not a real pdf")
    make_pdf(faq_dir / "good.pdf", "FAQ valido sobre renegociacao.")
    settings = make_settings(faq_dir)

    with (
        patch("app.ingestion.get_indexed_hash", AsyncMock(return_value=None)),
        patch("app.ingestion.embed_texts", AsyncMock(return_value=[[0.1, 0.2]])),
        patch("app.ingestion.delete_chunks_for_file", AsyncMock()),
        patch("app.ingestion.index_chunk", AsyncMock()),
        patch("app.ingestion.refresh_index", AsyncMock()),
    ):
        summary = await ingest_faq_directory(openai_client=object(), opensearch_client=object(), settings=settings)

    assert summary.files_failed == 1
    assert summary.files_indexed == 1


async def test_backend_unavailable_aborts_the_run(faq_dir: Path):
    make_pdf(faq_dir / "faq.pdf", "FAQ sobre renegociacao.")
    settings = make_settings(faq_dir)

    with (
        patch("app.ingestion.get_indexed_hash", AsyncMock(return_value=None)),
        patch("app.ingestion.embed_texts", AsyncMock(side_effect=KnowledgeBackendUnavailableError("down"))),
    ):
        with pytest.raises(KnowledgeBackendUnavailableError):
            await ingest_faq_directory(openai_client=object(), opensearch_client=object(), settings=settings)


async def test_partial_index_failure_rolls_back_instead_of_leaving_a_stuck_partial_state(faq_dir: Path):
    # Small chunk_size so the FAQ text splits into multiple chunks - the second index_chunk
    # call is the one that fails, after the first one already "succeeded".
    make_pdf(faq_dir / "faq.pdf", "Primeira parte do FAQ. " * 20 + "Segunda parte do FAQ. " * 20)
    settings = Settings(
        faq_pdf_dir=str(faq_dir), chunk_size=50, chunk_overlap=10, openai_api_key="sk-test"
    )

    with (
        patch("app.ingestion.get_indexed_hash", AsyncMock(return_value=None)),
        patch("app.ingestion.embed_texts", AsyncMock(return_value=[[0.1, 0.2]] * 10)),
        patch("app.ingestion.delete_chunks_for_file", AsyncMock()) as mock_delete,
        patch(
            "app.ingestion.index_chunk",
            AsyncMock(side_effect=[None, KnowledgeBackendUnavailableError("timed out")]),
        ),
        patch("app.ingestion.refresh_index", AsyncMock()) as mock_refresh,
    ):
        with pytest.raises(KnowledgeBackendUnavailableError):
            await ingest_faq_directory(openai_client=object(), opensearch_client=object(), settings=settings)

    # Rolled back the one chunk that did get written, so a later attempt sees no
    # existing chunks for this file (existing_hash=None) and retries it in full.
    mock_delete.assert_awaited_once()
    mock_refresh.assert_not_awaited()


async def test_directory_missing_returns_empty_summary(tmp_path: Path):
    settings = make_settings(tmp_path / "does-not-exist")

    summary = await ingest_faq_directory(openai_client=object(), opensearch_client=object(), settings=settings)

    assert summary.files_indexed == 0
    assert summary.files_skipped == 0
    assert summary.files_failed == 0
