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

configure_logging()
logger = logging.getLogger(__name__)

_tracer_provider = TracerProvider(resource=Resource.create({"service.name": "knowledge-service"}))
_tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=get_settings().otel_otlp_endpoint))
)
trace.set_tracer_provider(_tracer_provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.openai_client = build_openai_client(settings)
    app.state.opensearch_client = build_opensearch_client(settings)

    await ensure_index(app.state.opensearch_client, settings)

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not configured; skipping startup FAQ ingestion")
    else:
        try:
            summary = await ingest_faq_directory(
                app.state.openai_client, app.state.opensearch_client, settings
            )
            logger.info(
                "Startup ingestion: %s indexed, %s skipped, %s failed, %s chunks written",
                summary.files_indexed,
                summary.files_skipped,
                summary.files_failed,
                summary.chunks_written,
            )
        except KnowledgeBackendUnavailableError:
            logger.warning("Startup FAQ ingestion could not complete; continuing to serve requests")

    yield

    await app.state.opensearch_client.close()


app = FastAPI(title="knowledge-service", lifespan=lifespan)
app.add_middleware(CorrelationIdMiddleware)
FastAPIInstrumentor.instrument_app(app)


@app.exception_handler(RequestValidationError)
async def log_validation_errors(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Rejected %s: errors=%s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(KnowledgeBackendUnavailableError)
async def handle_backend_unavailable(request: Request, exc: KnowledgeBackendUnavailableError) -> JSONResponse:
    logger.error("Knowledge backend unavailable while handling %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


app.include_router(search_router)
app.include_router(admin_router)
