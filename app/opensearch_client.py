from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import OpenSearchException

from app.config import Settings
from app.errors import KnowledgeBackendUnavailableError

logger = logging.getLogger(__name__)


def build_opensearch_client(settings: Settings) -> AsyncOpenSearch:
    # max_retries=0 matters as much as the timeout itself: opensearch-py's default of 3
    # retries would otherwise silently multiply this into a ~9s wait on an unreachable
    # host before the 503 in app/main.py ever gets a chance to fire.
    return AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        use_ssl=settings.opensearch_url.startswith("https"),
        verify_certs=False,
        timeout=3,
        max_retries=0,
    )


async def ensure_index(client: AsyncOpenSearch, settings: Settings) -> None:
    try:
        exists = await client.indices.exists(index=settings.opensearch_index)
        if exists:
            return

        await client.indices.create(
            index=settings.opensearch_index,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "title": {"type": "text"},
                        "sourceFile": {"type": "keyword"},
                        "chunkIndex": {"type": "integer"},
                        "contentHash": {"type": "keyword"},
                        "createdAt": {"type": "date"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": settings.embedding_dimensions,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                # nmslib is deprecated for new indices from OpenSearch 3.0
                                # onward; lucene ships with core OpenSearch (no extra
                                # native-library plugin needed) and is the recommended
                                # default engine.
                                "engine": "lucene",
                            },
                        },
                    }
                },
            },
        )
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc


async def get_indexed_hash(client: AsyncOpenSearch, settings: Settings, source_file: str) -> str | None:
    """Returns the contentHash already indexed for source_file, or None if nothing is indexed for it."""
    try:
        result = await client.search(
            index=settings.opensearch_index,
            body={
                "size": 1,
                "query": {"term": {"sourceFile": source_file}},
                "_source": ["contentHash"],
            },
        )
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc

    hits = result["hits"]["hits"]
    return hits[0]["_source"]["contentHash"] if hits else None


async def count_indexed_chunks(client: AsyncOpenSearch, settings: Settings, source_file: str) -> int:
    """Counts documents already indexed for source_file.

    Used alongside get_indexed_hash to detect a *partial* prior ingestion: a hash match
    alone isn't proof the file finished indexing - a chunk write can time out client-side
    (see build_opensearch_client's timeout/max_retries) while still succeeding server-side,
    leaving fewer documents indexed than the file's current chunk count. Comparing counts
    catches that case so the file gets re-ingested instead of skipped forever.
    """
    try:
        result = await client.count(
            index=settings.opensearch_index, body={"query": {"term": {"sourceFile": source_file}}}
        )
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc

    return result["count"]


async def delete_chunks_for_file(client: AsyncOpenSearch, settings: Settings, source_file: str) -> None:
    try:
        await client.delete_by_query(
            index=settings.opensearch_index,
            body={"query": {"term": {"sourceFile": source_file}}},
            refresh=True,
        )
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc


async def index_chunk(
    client: AsyncOpenSearch,
    settings: Settings,
    *,
    source_file: str,
    title: str,
    chunk_index: int,
    text: str,
    content_hash: str,
    embedding: list[float],
) -> None:
    document = {
        "text": text,
        "title": title,
        "sourceFile": source_file,
        "chunkIndex": chunk_index,
        "contentHash": content_hash,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "embedding": embedding,
    }
    try:
        # No refresh here on purpose: forcing a segment refresh on every single chunk
        # serializes and slows down a real multi-chunk file enough to trip the 3s
        # connection timeout above on a perfectly healthy cluster. One explicit
        # refresh_index() call after a file's whole chunk set is written is enough for
        # it to become searchable.
        await client.index(index=settings.opensearch_index, body=document)
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc


async def refresh_index(client: AsyncOpenSearch, settings: Settings) -> None:
    try:
        await client.indices.refresh(index=settings.opensearch_index)
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc


async def knn_search(
    client: AsyncOpenSearch, settings: Settings, query_vector: list[float]
) -> list[dict[str, Any]]:
    try:
        result = await client.search(
            index=settings.opensearch_index,
            body={
                "size": settings.search_top_k,
                "query": {"knn": {"embedding": {"vector": query_vector, "k": settings.search_top_k}}},
            },
        )
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc

    return result["hits"]["hits"]
