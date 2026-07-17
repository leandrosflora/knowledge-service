from __future__ import annotations

from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    content: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ReindexResponse(BaseModel):
    files_indexed: int
    files_skipped: int
    files_failed: int
    chunks_written: int
