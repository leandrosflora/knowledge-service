import pytest
import respx
from httpx import Response

from app.config import Settings
from app.embeddings import build_openai_client, embed_query, embed_texts
from app.errors import KnowledgeBackendUnavailableError


def make_settings(api_key: str = "sk-test") -> Settings:
    return Settings(openai_api_key=api_key, embedding_model="text-embedding-3-small")


def embeddings_response(vectors: list[list[float]]) -> Response:
    return Response(
        200,
        json={
            "data": [{"embedding": v, "index": i, "object": "embedding"} for i, v in enumerate(vectors)],
            "model": "text-embedding-3-small",
            "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )


@respx.mock
async def test_embed_texts_returns_vectors_for_each_input():
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=embeddings_response([[0.1, 0.2], [0.3, 0.4]])
    )
    settings = make_settings()
    client = build_openai_client(settings)

    result = await embed_texts(client, settings, ["texto 1", "texto 2"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


@respx.mock
async def test_embed_query_returns_a_single_vector():
    respx.post("https://api.openai.com/v1/embeddings").mock(return_value=embeddings_response([[0.5, 0.6]]))
    settings = make_settings()
    client = build_openai_client(settings)

    result = await embed_query(client, settings, "pergunta")

    assert result == [0.5, 0.6]


async def test_embed_texts_missing_api_key_raises_without_calling_openai():
    settings = make_settings(api_key="")
    client = build_openai_client(settings)

    with pytest.raises(KnowledgeBackendUnavailableError):
        await embed_texts(client, settings, ["texto"])


@respx.mock
async def test_embed_texts_openai_failure_raises_mapped_exception():
    respx.post("https://api.openai.com/v1/embeddings").mock(return_value=Response(503))
    settings = make_settings()
    client = build_openai_client(settings)

    with pytest.raises(KnowledgeBackendUnavailableError):
        await embed_texts(client, settings, ["texto"])
