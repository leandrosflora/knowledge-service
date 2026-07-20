from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport

import app.main as main_module
from app.config import Settings
from app.dependencies import get_openai_client, get_opensearch_client, get_settings_dep
from app.ingestion import IngestionSummary
from app.main import app

TENANT_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
async def client():
    app.dependency_overrides[get_openai_client] = lambda: object()
    app.dependency_overrides[get_opensearch_client] = lambda: object()
    # get_settings_dep normally resolves from app.state.settings, only populated by the
    # lifespan handler - ASGITransport doesn't run lifespan events here, so it's overridden
    # directly instead (this is what the handler's own `settings` parameter receives).
    app.dependency_overrides[get_settings_dep] = lambda: Settings()
    # PlatformMiddleware is added with the module-level `settings` singleton at app-construction
    # time (see app/main.py), not resolved per-request via FastAPI DI - overriding
    # get_settings_dep has no effect on it. Mutating that same singleton in place (rather than
    # signing a real JWT) mirrors how conversation-orchestrator's WebApplicationFactory tests
    # bypass internal auth; PlatformMiddleware still requires and validates the X-Tenant-Id
    # header either way, so tenant scoping is still exercised end to end.
    original_auth_enabled = main_module.settings.internal_auth_enabled
    main_module.settings.internal_auth_enabled = False
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-Tenant-Id": TENANT_ID}
    ) as ac:
        yield ac
    app.dependency_overrides.clear()
    main_module.settings.internal_auth_enabled = original_auth_enabled


async def test_reindex_returns_ingestion_summary(client: httpx.AsyncClient):
    summary = IngestionSummary(files_indexed=1, files_skipped=2, files_failed=0, chunks_written=5)
    with (
        patch("app.api.admin.ensure_index", AsyncMock()),
        patch("app.api.admin.ingest_faq_directory", AsyncMock(return_value=summary)),
    ):
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

    with (
        patch("app.api.admin.ensure_index", AsyncMock()),
        patch(
            "app.api.admin.ingest_faq_directory", AsyncMock(side_effect=[first_summary, second_summary])
        ),
    ):
        first_response = await client.post("/admin/reindex")
        second_response = await client.post("/admin/reindex")

    assert first_response.json()["files_indexed"] == 1
    assert second_response.json()["files_indexed"] == 0
    assert second_response.json()["files_skipped"] == 1
