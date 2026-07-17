from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport

from app.config import Settings
from app.dependencies import get_openai_client, get_opensearch_client, get_settings_dep
from app.ingestion import IngestionSummary
from app.main import app


@pytest.fixture
async def client():
    app.dependency_overrides[get_openai_client] = lambda: object()
    app.dependency_overrides[get_opensearch_client] = lambda: object()
    app.dependency_overrides[get_settings_dep] = lambda: Settings()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_reindex_returns_ingestion_summary(client: httpx.AsyncClient):
    summary = IngestionSummary(files_indexed=1, files_skipped=2, files_failed=0, chunks_written=5)
    with patch("app.api.admin.ingest_faq_directory", AsyncMock(return_value=summary)):
        response = await client.post("/admin/reindex")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "files_indexed": 1,
        "files_skipped": 2,
        "files_failed": 0,
        "chunks_written": 5,
    }


async def test_reindex_is_idempotent_on_second_call(client: httpx.AsyncClient):
    first_summary = IngestionSummary(files_indexed=1, files_skipped=0, files_failed=0, chunks_written=3)
    second_summary = IngestionSummary(files_indexed=0, files_skipped=1, files_failed=0, chunks_written=0)

    with patch(
        "app.api.admin.ingest_faq_directory", AsyncMock(side_effect=[first_summary, second_summary])
    ):
        first_response = await client.post("/admin/reindex")
        second_response = await client.post("/admin/reindex")

    assert first_response.json()["files_indexed"] == 1
    assert second_response.json()["files_indexed"] == 0
    assert second_response.json()["files_skipped"] == 1
