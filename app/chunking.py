from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    text: str


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Splits text into fixed-size, overlapping character windows.

    Empty/whitespace-only text produces no chunks. The final chunk is whatever
    remains (shorter than chunk_size is expected and fine).
    """
    stripped = text.strip()
    if not stripped:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(stripped)

    while start < length:
        end = min(start + chunk_size, length)
        piece = stripped[start:end].strip()
        if piece:
            chunks.append(Chunk(index=index, text=piece))
            index += 1
        if end == length:
            break
        start += step

    return chunks
