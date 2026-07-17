from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    # Dimension of OpenAI's text-embedding-3-small - must match the knn_vector
    # mapping created in opensearch_client.py.
    embedding_dimensions: int = 1536

    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "faq_chunks"

    faq_pdf_dir: str = "data/faq_pdfs"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    search_top_k: int = 3
    min_relevance_score: float = 0.70

    otel_otlp_endpoint: str = "http://localhost:4317"


def get_settings() -> Settings:
    return Settings()
