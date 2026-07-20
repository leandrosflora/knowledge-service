from unittest.mock import AsyncMock, MagicMock

import pytest
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from app.config import Settings
from app.errors import KnowledgeBackendUnavailableError
from app.opensearch_client import (
    count_indexed_chunks,
    delete_chunks_for_file,
    ensure_index,
    get_indexed_hash,
    index_chunk,
    knn_search,
    refresh_index,
)

TENANT_ID = "00000000-0000-0000-0000-000000000001"


def make_settings() -> Settings:
    return Settings(opensearch_index_prefix="faq_chunks", embedding_dimensions=3, search_top_k=2)


def make_client() -> MagicMock:
    client = MagicMock()
    client.indices = MagicMock()
    client.indices.exists = AsyncMock()
    client.indices.create = AsyncMock()
    client.indices.refresh = AsyncMock()
    client.search = AsyncMock()
    client.count = AsyncMock()
    client.index = AsyncMock()
    client.delete_by_query = AsyncMock()
    return client


async def test_ensure_index_creates_when_absent():
    client = make_client()
    client.indices.exists.return_value = False

    await ensure_index(client, make_settings(), TENANT_ID)

    client.indices.create.assert_awaited_once()


async def test_ensure_index_is_a_noop_when_already_present():
    client = make_client()
    client.indices.exists.return_value = True

    await ensure_index(client, make_settings(), TENANT_ID)

    client.indices.create.assert_not_awaited()


async def test_ensure_index_connection_failure_raises_mapped_exception():
    client = make_client()
    client.indices.exists.side_effect = OpenSearchConnectionError("unreachable")

    with pytest.raises(KnowledgeBackendUnavailableError):
        await ensure_index(client, make_settings(), TENANT_ID)


async def test_get_indexed_hash_returns_none_when_no_match():
    client = make_client()
    client.search.return_value = {"hits": {"hits": []}}

    result = await get_indexed_hash(client, make_settings(), TENANT_ID, "faq.pdf")

    assert result is None


async def test_get_indexed_hash_returns_hash_when_present():
    client = make_client()
    client.search.return_value = {"hits": {"hits": [{"_source": {"contentHash": "abc123"}}]}}

    result = await get_indexed_hash(client, make_settings(), TENANT_ID, "faq.pdf")

    assert result == "abc123"


async def test_count_indexed_chunks_returns_count():
    client = make_client()
    client.count.return_value = {"count": 5}

    result = await count_indexed_chunks(client, make_settings(), TENANT_ID, "faq.pdf")

    assert result == 5


async def test_count_indexed_chunks_connection_failure_raises_mapped_exception():
    client = make_client()
    client.count.side_effect = OpenSearchConnectionError("unreachable")

    with pytest.raises(KnowledgeBackendUnavailableError):
        await count_indexed_chunks(client, make_settings(), TENANT_ID, "faq.pdf")


async def test_delete_chunks_for_file_calls_delete_by_query():
    client = make_client()

    await delete_chunks_for_file(client, make_settings(), TENANT_ID, "faq.pdf")

    client.delete_by_query.assert_awaited_once()


async def test_index_chunk_calls_index_with_document_fields():
    client = make_client()

    await index_chunk(
        client,
        make_settings(),
        TENANT_ID,
        source_file="faq.pdf",
        title="faq",
        chunk_index=0,
        text="conteudo",
        content_hash="abc123",
        embedding=[0.1, 0.2, 0.3],
    )

    _, kwargs = client.index.call_args
    document = kwargs["body"]
    assert document["sourceFile"] == "faq.pdf"
    assert document["contentHash"] == "abc123"
    assert document["embedding"] == [0.1, 0.2, 0.3]
    # No per-document refresh: forcing a segment refresh on every chunk is what made a
    # real multi-chunk file slow enough to trip the client's own connection timeout.
    assert "refresh" not in kwargs


async def test_refresh_index_calls_indices_refresh():
    client = make_client()

    await refresh_index(client, make_settings(), TENANT_ID)

    client.indices.refresh.assert_awaited_once_with(index=f"faq_chunks-{TENANT_ID}")


async def test_refresh_index_connection_failure_raises_mapped_exception():
    client = make_client()
    client.indices.refresh.side_effect = OpenSearchConnectionError("unreachable")

    with pytest.raises(KnowledgeBackendUnavailableError):
        await refresh_index(client, make_settings(), TENANT_ID)


async def test_knn_search_returns_hits():
    client = make_client()
    client.search.return_value = {"hits": {"hits": [{"_score": 0.9, "_source": {}}]}}

    hits = await knn_search(client, make_settings(), TENANT_ID, [0.1, 0.2, 0.3])

    assert hits == [{"_score": 0.9, "_source": {}}]


async def test_knn_search_connection_failure_raises_mapped_exception():
    client = make_client()
    client.search.side_effect = OpenSearchConnectionError("unreachable")

    with pytest.raises(KnowledgeBackendUnavailableError):
        await knn_search(client, make_settings(), TENANT_ID, [0.1, 0.2, 0.3])
