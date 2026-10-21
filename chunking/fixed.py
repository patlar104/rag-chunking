"""Strategy 01: fixed-size chunking (fast character or token slicing)."""
from __future__ import annotations

from .base import Chunk, token_spans


class FixedSizeChunker:
    """Slice text every `chunk_size` tokens (or characters).

    Fast and predictable, but boundaries ignore meaning, so it happily cuts
    mid-sentence. `overlap` turns it into the sliding-window variant from the
    reel: windows advance by `chunk_size - overlap`, so neighbours share text.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 0, unit: str = "tokens"):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        if unit not in ("tokens", "chars"):
            raise ValueError("unit must be 'tokens' or 'chars'")
        self.chunk_size, self.overlap, self.unit = chunk_size, overlap, unit

    def split(self, text: str) -> list[Chunk]:
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        if self.unit == "chars":
            for i in range(0, len(text), step):
                piece = text[i : i + self.chunk_size]
                if piece.strip():
                    chunks.append(Chunk(piece, i, i + len(piece), {"strategy": "fixed"}))
                if i + self.chunk_size >= len(text):
                    break
            return chunks

        spans = token_spans(text)
        for i in range(0, len(spans), step):
            window = spans[i : i + self.chunk_size]
            a, b = window[0][0], window[-1][1]
            chunks.append(Chunk(text[a:b], a, b, {"strategy": "fixed", "first_token": i}))
            if i + self.chunk_size >= len(spans):
                break
        return chunks
