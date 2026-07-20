import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.api.admin import router as admin_router
from app.api.search import router as search_router
from app.config import get_settings
from app.embeddings import build_openai_client
from app.errors import KnowledgeBackendUnavailableError
from app.ingestion import ingest_faq_directory
from app.logging_setup import CorrelationIdMiddleware, configure_logging
from app.opensearch_client import build_opensearch_client, ensure_index
from app.platform import PlatformMiddleware, metrics_response

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

_tracer_provider = TracerProvider(
    resource=Resource.create({"service.name": settings.internal_auth_service_name})
)
_tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_otlp_endpoint))
)
trace.set_tracer_provider(_tracer_provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.openai_client = build_openai_client(settings)
    app.state.opensearch_client = build_opensearch_client(settings)

    try:
        await ensure_index(app.state.opensearch_client, settings, settings.default_tenant_id)
        if settings.openai_api_key:
            summary = await ingest_faq_directory(
                app.state.openai_client,
                app.state.opensearch_client,
                settings,
                settings.default_tenant_id,
            )
            logger.info(
                "Startup ingestion for tenant %s: %s indexed, %s skipped, %s failed, %s chunks",
                settings.default_tenant_id,
                summary.files_indexed,
                summary.files_skipped,
                summary.files_failed,
                summary.chunks_written,
            )
        else:
            logger.warning("OPENAI_API_KEY not configured; skipping startup FAQ ingestion")
    except KnowledgeBackendUnavailableError:
        logger.warning("Startup FAQ ingestion could not complete; readiness will remain false")

    yield
    await app.state.opensearch_client.close()


app = FastAPI(title="knowledge-service", lifespan=lifespan)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    PlatformMiddleware,
    settings=settings,
    public_paths=("/health/live", "/health/ready", "/metrics", "/docs", "/openapi.json", "/redoc"),
    tenant_required_paths=("/search", "/admin"),
)
FastAPIInstrumentor.instrument_app(app)


@app.exception_handler(RequestValidationError)
async def log_validation_errors(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Rejected %s: errors=%s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(KnowledgeBackendUnavailableError)
async def handle_backend_unavailable(request: Request, exc: KnowledgeBackendUnavailableError) -> JSONResponse:
    logger.error("Knowledge backend unavailable while handling %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health/live", include_in_schema=False)
async def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready", include_in_schema=False)
async def health_ready(request: Request) -> JSONResponse:
    failures: list[str] = []
    if settings.internal_auth_enabled:
        inbound_secret = settings.internal_auth_inbound_secrets.get("agent-runtime-renegotiation")
        if not inbound_secret or len(inbound_secret.encode("utf-8")) < 32:
            failures.append("internal_auth_inbound_secret_missing:agent-runtime-renegotiation")
    if not settings.openai_api_key:
        failures.append("openai_api_key_missing")
    try:
        await request.app.state.opensearch_client.cluster.health()
    except Exception:
        failures.append("opensearch_unavailable")

    return JSONResponse(
        {"status": "not_ready" if failures else "ready", "failures": failures},
        status_code=503 if failures else 200,
    )


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return metrics_response()


app.include_router(search_router)
app.include_router(admin_router)
