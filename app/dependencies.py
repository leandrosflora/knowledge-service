from __future__ import annotations

from fastapi import Request
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from app.config import Settings


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_openai_client(request: Request) -> AsyncOpenAI:
    return request.app.state.openai_client


def get_opensearch_client(request: Request) -> AsyncOpenSearch:
    return request.app.state.opensearch_client
