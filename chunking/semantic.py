"""Strategy 04: semantic chunking (cut when embedding distance spikes)."""
from __future__ import annotations

from typing import List, Optional

from . import events
from .base import Chunk, attach_headings, count_tokens, sentence_spans
from .embeddings import Embedder, HashingEmbedder, cosine


class SemanticChunker:
    """Embed each sentence (with a little neighbouring context), measure the
    cosine distance between consecutive sentences, and start a new chunk
    wherever that distance spikes above the `breakpoint_percentile`.

    Args:
        embed: any `list[str] -> list[vector]` function (default: offline hashing).
        breakpoint_percentile: distances above this percentile become cuts.
        buffer: sentences compared on each side of a candidate cut (smooths noise).
        min_sentences / max_tokens: guard rails so chunks are never absurd.
    """

    def __init__(self, embed: Optional[Embedder] = None, breakpoint_percentile: float = 80,
                 buffer: int = 2, min_sentences: int = 2, max_tokens: int = 400):
        self.embed = embed or HashingEmbedder()
        self.pct = breakpoint_percentile
        self.buffer = buffer
        self.min_sentences = min_sentences
        self.max_tokens = max_tokens

    def distances(self, sentences: List[str]) -> List[float]:
        """Distance at each gap i|i+1, comparing `buffer` sentences on the left
        with `buffer` sentences on the right (non-overlapping windows, like
        TextTiling). buffer=0 compares single sentences."""
        b = max(1, self.buffer)
        left = [" ".join(sentences[max(0, i - b + 1): i + 1]) for i in range(len(sentences) - 1)]
        right = [" ".join(sentences[i + 1: i + 1 + b]) for i in range(len(sentences) - 1)]
        vecs = self.embed(left + right)
        n = len(left)
        return [1 - cosine(vecs[i], vecs[n + i]) for i in range(n)]

    def split(self, text: str) -> List[Chunk]:
        spans = attach_headings(text, sentence_spans(text))
        if len(spans) < 2:
            return [Chunk(text.strip(), *spans[0], {"strategy": "semantic"})] if spans else []
        sentences = [text[a:b] for a, b in spans]
        dists = self.distances(sentences)
        threshold = _percentile(dists, self.pct)
        events.emit("semantic.distances", distances=[round(d, 4) for d in dists], threshold=round(threshold, 4),
                    gaps=[spans[i][1] for i in range(len(spans) - 1)], sentences=spans)

        chunks, group_start = [], 0
        for i in range(1, len(spans) + 1):
            at_end = i == len(spans)
            size = count_tokens(text[spans[group_start][0]: spans[i - 1][1]])
            spike = not at_end and dists[i - 1] > threshold and i - group_start >= self.min_sentences
            too_big = not at_end and size + count_tokens(sentences[i]) > self.max_tokens
            if at_end or spike or too_big:
                a, b = spans[group_start][0], spans[i - 1][1]
                chunks.append(Chunk(text[a:b], a, b, {
                    "strategy": "semantic",
                    "sentences": i - group_start,
                    "cut_reason": "end" if at_end else ("distance spike" if spike else "max_tokens"),
                    "cut_distance": None if at_end else round(dists[i - 1], 3),
                    "threshold": round(threshold, 3),
                }))
                group_start = i
        return chunks


def _percentile(values: List[float], pct: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
