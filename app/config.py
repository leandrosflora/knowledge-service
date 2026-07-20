from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    opensearch_url: str = "http://localhost:9200"
    opensearch_index_prefix: str = "faq_chunks"

    faq_pdf_dir: str = "data/faq_pdfs"
    default_tenant_id: str = "00000000-0000-0000-0000-000000000001"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    search_top_k: int = 3
    min_relevance_score: float = 0.70
    otel_otlp_endpoint: str = "http://localhost:4317"

    internal_auth_enabled: bool = True
    internal_auth_issuer: str = "conversational-ai-platform"
    internal_auth_service_name: str = "knowledge-service"
    internal_auth_outbound_secrets: dict[str, str] = {}
    internal_auth_inbound_secrets: dict[str, str] = {}
    internal_auth_token_ttl_seconds: int = 300


def get_settings() -> Settings:
    return Settings()
