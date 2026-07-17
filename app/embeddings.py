from __future__ import annotations

import logging

from openai import AsyncOpenAI, OpenAIError

from app.config import Settings
from app.errors import KnowledgeBackendUnavailableError

logger = logging.getLogger(__name__)


def build_openai_client(settings: Settings) -> AsyncOpenAI:
    # AsyncOpenAI itself refuses to construct with a falsy api_key (raises OpenAIError
    # immediately), which would crash app startup whenever no key is configured - before
    # embed_texts' own "not configured" check below ever runs. A placeholder satisfies
    # the SDK's construction-time check without ever being used for a real call: embed_texts
    # still raises before attempting one whenever the real settings.openai_api_key is empty.
    return AsyncOpenAI(api_key=settings.openai_api_key or "not-configured")


async def embed_texts(client: AsyncOpenAI, settings: Settings, texts: list[str]) -> list[list[float]]:
    if not settings.openai_api_key:
        raise KnowledgeBackendUnavailableError("OPENAI_API_KEY is not configured")

    try:
        response = await client.embeddings.create(model=settings.embedding_model, input=texts)
    except OpenAIError as exc:
        logger.warning("OpenAI embeddings call failed", exc_info=True)
        raise KnowledgeBackendUnavailableError("OpenAI embeddings API unavailable") from exc

    return [item.embedding for item in response.data]


async def embed_query(client: AsyncOpenAI, settings: Settings, query: str) -> list[float]:
    embeddings = await embed_texts(client, settings, [query])
    return embeddings[0]
