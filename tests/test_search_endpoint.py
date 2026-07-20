from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport

import app.main as main_module
from app.config import Settings
from app.dependencies import get_openai_client, get_opensearch_client, get_settings_dep
from app.errors import KnowledgeBackendUnavailableError
from app.main import app

TENANT_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
async def client():
    app.dependency_overrides[get_openai_client] = lambda: object()
    app.dependency_overrides[get_opensearch_client] = lambda: object()
    # get_settings_dep normally resolves from app.state.settings, only populated by the
    # lifespan handler - ASGITransport doesn't run lifespan events here, so it's overridden
    # directly instead (this is what the handler's own `settings` parameter receives, distinct
    # from the module-level singleton PlatformMiddleware holds - see below).
    app.dependency_overrides[get_settings_dep] = lambda: Settings(min_relevance_score=0.70)
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


async def test_search_returns_results_above_threshold(client: httpx.AsyncClient):
    hits = [
        {"_score": 0.85, "_source": {"title": "FAQ 1", "text": "resposta relevante"}},
        {"_score": 0.40, "_source": {"title": "FAQ 2", "text": "resposta irrelevante"}},
    ]
    with (
        patch("app.api.search.ensure_index", AsyncMock()),
        patch("app.api.search.embed_query", AsyncMock(return_value=[0.1, 0.2, 0.3])),
        patch("app.api.search.knn_search", AsyncMock(return_value=hits)),
    ):
        response = await client.get("/search", params={"query": "como renegociar?"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["title"] == "FAQ 1"
    assert body["results"][0]["score"] == 0.85


async def test_search_no_qualifying_hits_returns_empty_results(client: httpx.AsyncClient):
    hits = [{"_score": 0.2, "_source": {"title": "FAQ", "text": "pouco relevante"}}]
    with (
        patch("app.api.search.ensure_index", AsyncMock()),
        patch("app.api.search.embed_query", AsyncMock(return_value=[0.1, 0.2, 0.3])),
        patch("app.api.search.knn_search", AsyncMock(return_value=hits)),
    ):
        response = await client.get("/search", params={"query": "pergunta sem resposta"})

    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_search_empty_index_returns_empty_results(client: httpx.AsyncClient):
    with (
        patch("app.api.search.ensure_index", AsyncMock()),
        patch("app.api.search.embed_query", AsyncMock(return_value=[0.1, 0.2, 0.3])),
        patch("app.api.search.knn_search", AsyncMock(return_value=[])),
    ):
        response = await client.get("/search", params={"query": "qualquer coisa"})

    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_search_missing_query_returns_422(client: httpx.AsyncClient):
    response = await client.get("/search")

    assert response.status_code == 422


async def test_search_opensearch_unavailable_returns_503(client: httpx.AsyncClient):
    with (
        patch("app.api.search.ensure_index", AsyncMock()),
        patch("app.api.search.embed_query", AsyncMock(return_value=[0.1, 0.2, 0.3])),
        patch("app.api.search.knn_search", AsyncMock(side_effect=KnowledgeBackendUnavailableError("down"))),
    ):
        response = await client.get("/search", params={"query": "qualquer coisa"})

    assert response.status_code == 503


async def test_search_embeddings_unavailable_returns_503(client: httpx.AsyncClient):
    with (
        patch("app.api.search.ensure_index", AsyncMock()),
        patch(
            "app.api.search.embed_query", AsyncMock(side_effect=KnowledgeBackendUnavailableError("no key"))
        ),
    ):
        response = await client.get("/search", params={"query": "qualquer coisa"})

    assert response.status_code == 503
