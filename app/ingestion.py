from __future__ import annotations

import logging
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
) -> IngestionSummary:
    summary = IngestionSummary()
    faq_dir = Path(settings.faq_pdf_dir)

    if not faq_dir.is_dir():
        logger.info("FAQ PDF directory %s does not exist; nothing to ingest", faq_dir)
        return summary

    for pdf_path in sorted(faq_dir.glob("*.pdf")):
        try:
            await _ingest_file(openai_client, opensearch_client, settings, pdf_path, summary)
        except KnowledgeBackendUnavailableError:
            # OpenSearch/OpenAI being down affects every remaining file identically -
            # one clear error beats N identical per-file "failed" entries.
            logger.error(
                "Knowledge backend unavailable; aborting ingestion run after %s file(s)",
                pdf_path.name,
                exc_info=True,
            )
            raise
        except Exception:
            logger.warning("Failed to ingest %s; skipping", pdf_path.name, exc_info=True)
            summary.files_failed += 1

    return summary


async def _ingest_file(
    openai_client: AsyncOpenAI,
    opensearch_client: AsyncOpenSearch,
    settings: Settings,
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

    existing_hash = await get_indexed_hash(opensearch_client, settings, pdf_path.name)
    if existing_hash == extracted.content_hash:
        # A hash match alone doesn't prove the file finished indexing last time - a
        # chunk write can time out client-side while still landing server-side (see
        # opensearch_client.count_indexed_chunks), leaving fewer docs than len(chunks).
        # Only skip when the count also matches; otherwise fall through and re-ingest.
        existing_count = await count_indexed_chunks(opensearch_client, settings, pdf_path.name)
        if existing_count == len(chunks):
            summary.files_skipped += 1
            return
        logger.warning(
            "%s has a matching contentHash but only %s/%s chunks indexed; re-ingesting",
            pdf_path.name,
            existing_count,
            len(chunks),
        )

    embeddings = await embed_texts(openai_client, settings, [chunk.text for chunk in chunks])

    if existing_hash is not None:
        await delete_chunks_for_file(opensearch_client, settings, pdf_path.name)

    written = 0
    try:
        for chunk, embedding in zip(chunks, embeddings):
            await index_chunk(
                opensearch_client,
                settings,
                source_file=pdf_path.name,
                title=pdf_path.stem,
                chunk_index=chunk.index,
                text=chunk.text,
                content_hash=extracted.content_hash,
                embedding=embedding,
            )
            written += 1
    except KnowledgeBackendUnavailableError:
        # Without this rollback, a failure partway through would leave some chunks
        # indexed with the file's *current* contentHash - get_indexed_hash would then
        # see a "match" and skip this file forever, never finishing the ingestion that
        # actually failed. Better to leave nothing indexed for it than a stuck partial
        # state that looks complete.
        if written:
            try:
                await delete_chunks_for_file(opensearch_client, settings, pdf_path.name)
            except KnowledgeBackendUnavailableError:
                logger.error(
                    "Failed to roll back partial chunks for %s after an ingestion failure",
                    pdf_path.name,
                    exc_info=True,
                )
        raise

    await refresh_index(opensearch_client, settings)
    summary.chunks_written += written
    summary.files_indexed += 1
