from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import OpenSearchException

from app.config import Settings
from app.errors import KnowledgeBackendUnavailableError
from app.platform import tenant_index_name


def build_opensearch_client(settings: Settings) -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        use_ssl=settings.opensearch_url.startswith("https"),
        verify_certs=settings.opensearch_url.startswith("https"),
        timeout=3,
        max_retries=0,
    )


async def ensure_index(client: AsyncOpenSearch, settings: Settings, tenant_id: str) -> None:
    index_name = tenant_index_name(settings, tenant_id)
    try:
        if await client.indices.exists(index=index_name):
            return
        await client.indices.create(
            index=index_name,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "tenantId": {"type": "keyword"},
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
                                "engine": "lucene",
                            },
                        },
                    }
                },
            },
        )
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc


async def get_indexed_hash(
    client: AsyncOpenSearch,
    settings: Settings,
    tenant_id: str,
    source_file: str,
) -> str | None:
    try:
        result = await client.search(
            index=tenant_index_name(settings, tenant_id),
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


async def count_indexed_chunks(
    client: AsyncOpenSearch,
    settings: Settings,
    tenant_id: str,
    source_file: str,
) -> int:
    try:
        result = await client.count(
            index=tenant_index_name(settings, tenant_id),
            body={"query": {"term": {"sourceFile": source_file}}},
        )
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc
    return result["count"]


async def delete_chunks_for_file(
    client: AsyncOpenSearch,
    settings: Settings,
    tenant_id: str,
    source_file: str,
) -> None:
    try:
        await client.delete_by_query(
            index=tenant_index_name(settings, tenant_id),
            body={"query": {"term": {"sourceFile": source_file}}},
            refresh=True,
        )
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc


async def index_chunk(
    client: AsyncOpenSearch,
    settings: Settings,
    tenant_id: str,
    *,
    source_file: str,
    title: str,
    chunk_index: int,
    text: str,
    content_hash: str,
    embedding: list[float],
) -> None:
    document = {
        "tenantId": tenant_id,
        "text": text,
        "title": title,
        "sourceFile": source_file,
        "chunkIndex": chunk_index,
        "contentHash": content_hash,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "embedding": embedding,
    }
    try:
        await client.index(index=tenant_index_name(settings, tenant_id), body=document)
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc


async def refresh_index(client: AsyncOpenSearch, settings: Settings, tenant_id: str) -> None:
    try:
        await client.indices.refresh(index=tenant_index_name(settings, tenant_id))
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc


async def knn_search(
    client: AsyncOpenSearch,
    settings: Settings,
    tenant_id: str,
    query_vector: list[float],
) -> list[dict[str, Any]]:
    try:
        result = await client.search(
            index=tenant_index_name(settings, tenant_id),
            body={
                "size": settings.search_top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_vector,
                            "k": settings.search_top_k,
                        }
                    }
                },
            },
        )
    except OpenSearchException as exc:
        raise KnowledgeBackendUnavailableError("OpenSearch unavailable") from exc
    return result["hits"]["hits"]
