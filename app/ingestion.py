from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from app.chunking import chunk_text
from app.config import Settings
from app.embeddings import embed_texts
from app.errors import KnowledgeBackendUnavailableError
from app.opensearch_client import (
    count_indexed_chunks,
    delete_chunks_for_file,
    get_indexed_hash,
    index_chunk,
    refresh_index,
)
from app.pdf_extraction import PdfExtractionError, extract_pdf

logger = logging.getLogger(__name__)


@dataclass
class IngestionSummary:
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_written: int = 0


async def ingest_faq_directory(
    openai_client: AsyncOpenAI,
    opensearch_client: AsyncOpenSearch,
    settings: Settings,
    tenant_id: str,
) -> IngestionSummary:
    summary = IngestionSummary()
    faq_dir = _tenant_faq_directory(settings, tenant_id)

    if not faq_dir.is_dir():
        logger.info("FAQ directory for tenant %s does not exist; nothing to ingest", tenant_id)
        return summary

    for pdf_path in sorted(faq_dir.glob("*.pdf")):
        try:
            await _ingest_file(
                openai_client,
                opensearch_client,
                settings,
                tenant_id,
                pdf_path,
                summary,
            )
        except KnowledgeBackendUnavailableError:
            logger.error(
                "Knowledge backend unavailable for tenant %s; aborting ingestion at %s",
                tenant_id,
                pdf_path.name,
                exc_info=True,
            )
            raise
        except Exception:
            logger.warning(
                "Failed to ingest %s for tenant %s; skipping",
                pdf_path.name,
                tenant_id,
                exc_info=True,
            )
            summary.files_failed += 1

    return summary


async def _ingest_file(
    openai_client: AsyncOpenAI,
    opensearch_client: AsyncOpenSearch,
    settings: Settings,
    tenant_id: str,
    pdf_path: Path,
    summary: IngestionSummary,
) -> None:
    try:
        extracted = extract_pdf(pdf_path)
    except PdfExtractionError:
        logger.warning("Could not extract text from %s; skipping", pdf_path.name, exc_info=True)
        summary.files_failed += 1
        return

    chunks = chunk_text(extracted.text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        logger.warning("No extractable text in %s; skipping", pdf_path.name)
        summary.files_failed += 1
        return

    existing_hash = await get_indexed_hash(
        opensearch_client,
        settings,
        tenant_id,
        pdf_path.name,
    )
    if existing_hash == extracted.content_hash:
        existing_count = await count_indexed_chunks(
            opensearch_client,
            settings,
            tenant_id,
            pdf_path.name,
        )
        if existing_count == len(chunks):
            summary.files_skipped += 1
            return

    embeddings = await embed_texts(openai_client, settings, [chunk.text for chunk in chunks])

    if existing_hash is not None:
        await delete_chunks_for_file(
            opensearch_client,
            settings,
            tenant_id,
            pdf_path.name,
        )

    written = 0
    try:
        for chunk, embedding in zip(chunks, embeddings):
            await index_chunk(
                opensearch_client,
                settings,
                tenant_id,
                source_file=pdf_path.name,
                title=pdf_path.stem,
                chunk_index=chunk.index,
                text=chunk.text,
                content_hash=extracted.content_hash,
                embedding=embedding,
            )
            written += 1
    except KnowledgeBackendUnavailableError:
        if written:
            try:
                await delete_chunks_for_file(
                    opensearch_client,
                    settings,
                    tenant_id,
                    pdf_path.name,
                )
            except KnowledgeBackendUnavailableError:
                logger.error(
                    "Failed to roll back partial chunks for %s tenant %s",
                    pdf_path.name,
                    tenant_id,
                    exc_info=True,
                )
        raise

    await refresh_index(opensearch_client, settings, tenant_id)
    summary.chunks_written += written
    summary.files_indexed += 1


def _tenant_faq_directory(settings: Settings, tenant_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tenant_id):
        raise ValueError("Tenant ID contains unsupported path characters")

    root = Path(settings.faq_pdf_dir)
    tenant_dir = root / tenant_id
    if tenant_dir.is_dir():
        return tenant_dir

    # Backward-compatible migration path: the original root directory belongs only to the
    # configured default tenant. Other tenants never fall back to shared documents.
    if tenant_id == settings.default_tenant_id:
        return root
    return tenant_dir
