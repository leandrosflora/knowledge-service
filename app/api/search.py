from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch
from prometheus_client import Counter, Histogram

from app.config import Settings
from app.dependencies import get_openai_client, get_opensearch_client, get_settings_dep
from app.embeddings import embed_query
from app.models import SearchResponse, SearchResult
from app.opensearch_client import ensure_index, knn_search
from app.platform import current_tenant_id

router = APIRouter(tags=["search"])

SEARCHES = Counter(
    "knowledge_searches_total",
    "Knowledge searches executed.",
    ["outcome"],
)
RESULTS = Histogram(
    "knowledge_search_result_count",
    "Number of relevant chunks returned per search.",
)


@router.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
    opensearch_client: AsyncOpenSearch = Depends(get_opensearch_client),
    settings: Settings = Depends(get_settings_dep),
) -> SearchResponse:
    tenant_id = current_tenant_id()
    await ensure_index(opensearch_client, settings, tenant_id)
    query_vector = await embed_query(openai_client, settings, query)
    hits = await knn_search(opensearch_client, settings, tenant_id, query_vector)

    results = [
        SearchResult(title=hit["_source"]["title"], content=hit["_source"]["text"], score=hit["_score"])
        for hit in hits
        if hit["_score"] >= settings.min_relevance_score
    ]
    results.sort(key=lambda result: result.score, reverse=True)
    SEARCHES.labels("results" if results else "empty").inc()
    RESULTS.observe(len(results))
    return SearchResponse(results=results)
