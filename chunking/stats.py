"""Quick diagnostics for the two failure modes in the reel.

Problem #1 - too small  = lost context  -> look at `mid_sentence_cuts` and `tiny_chunks`
Problem #2 - too large  = embedding noise -> look at `max_tokens` and `oversized_chunks`
"""
from __future__ import annotations

from statistics import mean, median

from .base import Chunk, sentence_spans


def chunk_stats(text: str, chunks: list[Chunk], tiny: int = 15, oversized: int = 400) -> dict:
    if not chunks:
        return {"chunks": 0}
    sizes = [c.tokens for c in chunks]
    sentence_ends = {b for _, b in sentence_spans(text)} | {len(text.rstrip())}
    sentence_starts = {a for a, _ in sentence_spans(text)}
    real = [c for c in chunks if c.start >= 0 and text[c.start:c.end] == c.text]
    cuts = sum(1 for c in real if c.end not in sentence_ends) + \
        sum(1 for c in real[1:] if c.start not in sentence_starts and not c.metadata.get("overlap_tokens"))
    return {
        "chunks": len(chunks),
        "avg_tokens": round(mean(sizes), 1),
        "median_tokens": median(sizes),
        "min_tokens": min(sizes),
        "max_tokens": max(sizes),
        "tiny_chunks": sum(s < tiny for s in sizes),
        "oversized_chunks": sum(s > oversized for s in sizes),
        "mid_sentence_cuts": cuts,
    }
