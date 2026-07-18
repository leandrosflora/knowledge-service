from __future__ import annotations

from fastapi import APIRouter, Depends
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch
from prometheus_client import Counter

from app.config import Settings
from app.dependencies import get_openai_client, get_opensearch_client, get_settings_dep
from app.ingestion import ingest_faq_directory
from app.models import ReindexResponse
from app.opensearch_client import ensure_index
from app.platform import current_tenant_id

router = APIRouter(prefix="/admin", tags=["admin"])
REINDEX_RUNS = Counter(
    "knowledge_reindex_runs_total",
    "Tenant-scoped FAQ reindex runs.",
    ["outcome"],
)


@router.post("/reindex", response_model=ReindexResponse)
async def reindex(
    openai_client: AsyncOpenAI = Depends(get_openai_client),
    opensearch_client: AsyncOpenSearch = Depends(get_opensearch_client),
    settings: Settings = Depends(get_settings_dep),
) -> ReindexResponse:
    tenant_id = current_tenant_id()
    try:
        await ensure_index(opensearch_client, settings, tenant_id)
        summary = await ingest_faq_directory(
            openai_client,
            opensearch_client,
            settings,
            tenant_id,
        )
        REINDEX_RUNS.labels("success").inc()
    except Exception:
        REINDEX_RUNS.labels("error").inc()
        raise

    return ReindexResponse(
        files_indexed=summary.files_indexed,
        files_skipped=summary.files_skipped,
        files_failed=summary.files_failed,
        chunks_written=summary.chunks_written,
    )
