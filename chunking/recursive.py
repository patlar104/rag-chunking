"""Strategy 02: recursive chunking (paragraphs -> sentences -> words)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .base import Chunk, LengthFn, Span, count_tokens, spans_to_chunks

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
_LEVELS = {
    "\n\n": "paragraph",
    "\n": "line",
    ". ": "sentence",
    "? ": "sentence",
    "! ": "sentence",
    "; ": "clause",
    ", ": "clause",
    " ": "word",
    "": "character",
}


class RecursiveChunker:
    """Try the coarsest separator first; only drop to a finer one when a piece
    is still too big. Neighbouring small pieces are merged back up to
    `chunk_size`, so you get chunks that respect paragraphs where possible,
    sentences when needed, and words only as a last resort.
    """

    def __init__(
        self,
        chunk_size: int = 200,
        separators: Sequence[str] | None = None,
        length_fn: LengthFn = count_tokens,
    ):
        self.chunk_size = chunk_size
        self.separators = list(separators or DEFAULT_SEPARATORS)
        self.length = length_fn

    def split(self, text: str) -> list[Chunk]:
        chunks = []
        for (a, b), level in self._split(text, 0, len(text), self.separators, "document"):
            chunks.extend(spans_to_chunks(text, [(a, b)], strategy="recursive", split_level=level))
        return chunks

    # -- internals -------------------------------------------------------
    def _pieces(self, text: str, start: int, end: int, sep: str) -> list[Span]:
        """Split text[start:end] on `sep`, keeping the separator on the left piece."""
        if sep == "":
            return [(i, i + 1) for i in range(start, end)]
        pieces, pos = [], start
        for m in re.finditer(re.escape(sep), text[start:end]):
            cut = start + m.end()
            pieces.append((pos, cut))
            pos = cut
        if pos < end:
            pieces.append((pos, end))
        return pieces

    def _split(self, text, start, end, seps, level):
        if self.length(text[start:end]) <= self.chunk_size or not seps:
            return [((start, end), level)]
        segment = text[start:end]
        i = next((k for k, s in enumerate(seps) if s == "" or s in segment), len(seps) - 1)
        sep, finer = seps[i], seps[i + 1 :]
        level = _LEVELS.get(sep, f"split on {sep!r}")

        out, cur = [], None
        for ps, pe in self._pieces(text, start, end, sep):
            if self.length(text[ps:pe]) > self.chunk_size:
                if cur:
                    out.append((cur, level))
                    cur = None
                out.extend(self._split(text, ps, pe, finer, level))
            elif cur is None:
                cur = (ps, pe)
            elif self.length(text[cur[0] : pe]) <= self.chunk_size:
                cur = (cur[0], pe)
            else:
                out.append((cur, level))
                cur = (ps, pe)
        if cur:
            out.append((cur, level))
        return out
