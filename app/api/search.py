from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from app.config import Settings
from app.dependencies import get_openai_client, get_opensearch_client, get_settings_dep
from app.embeddings import embed_query
from app.models import SearchResponse, SearchResult
from app.opensearch_client import knn_search

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
    opensearch_client: AsyncOpenSearch = Depends(get_opensearch_client),
    settings: Settings = Depends(get_settings_dep),
) -> SearchResponse:
    query_vector = await embed_query(openai_client, settings, query)
    hits = await knn_search(opensearch_client, settings, query_vector)

    results = [
        SearchResult(title=hit["_source"]["title"], content=hit["_source"]["text"], score=hit["_score"])
        for hit in hits
        if hit["_score"] >= settings.min_relevance_score
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return SearchResponse(results=results)
