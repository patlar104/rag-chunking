"""Shared building blocks: the Chunk type, tokenising, sentence splitting, overlap."""
from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

Span = tuple[int, int]

# A deliberately simple "token": a word or a single punctuation mark.
# Swap in tiktoken or a HF tokenizer via `length_fn` if you need exact model counts.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

# Sentence end = . ! ? (optionally followed by quotes/brackets) then whitespace.
_SENT_END_RE = re.compile(r"[.!?][\"')\]]*\s+|\n\s*\n")


@dataclass
class Chunk:
    """One retrievable unit of text.

    `start`/`end` are character offsets into the original document, so you can
    always trace a chunk back to its source. Synthetic chunks (e.g. LLM-written
    propositions) point at the span they were derived from.
    """

    text: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)

    def __repr__(self) -> str:  # short, readable repr for the REPL
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk({self.start}:{self.end}, {self.tokens} tok, {preview!r}...)"


def token_spans(text: str) -> list[Span]:
    return [m.span() for m in _TOKEN_RE.finditer(text)]


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Split text[start:end] into sentence spans (absolute offsets, whitespace trimmed)."""
    end = len(text) if end is None else end
    spans: list[Span] = []
    pos = start
    for m in _SENT_END_RE.finditer(text, start, end):
        cut = m.start() + len(m.group(0).rstrip())
        spans.append((pos, cut))
        pos = m.end()
    spans.append((pos, end))
    return [s for s in (strip_span(text, a, b) for a, b in spans) if s[1] > s[0]]


def attach_headings(text: str, spans: list[Span]) -> list[Span]:
    """Glue a lone Markdown heading onto the sentence after it, so a heading
    never ends up as the last line of the previous chunk."""
    out: list[Span] = []
    pending = None
    for a, b in spans:
        is_heading = text[a:b].lstrip().startswith("#") and "\n" not in text[a:b].strip()
        if pending is not None:
            a = pending
            pending = None
        if is_heading:
            pending = a
            continue
        out.append((a, b))
    if pending is not None:
        out.append((pending, spans[-1][1]))
    return out


def strip_span(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def spans_to_chunks(text: str, spans: Iterable[Span], **metadata) -> list[Chunk]:
    chunks = []
    for a, b in spans:
        a, b = strip_span(text, a, b)
        if b > a:
            chunks.append(Chunk(text[a:b], a, b, dict(metadata)))
    return chunks


def with_overlap(text: str, chunks: list[Chunk], overlap_tokens: int = 30) -> list[Chunk]:
    """The reel's "practical fix": slide each chunk's start back by N tokens.

    Works with any chunker that returns real character offsets. Each chunk
    (except the first) is extended to begin `overlap_tokens` tokens earlier,
    so a fact cut at a boundary still appears whole in at least one chunk.
    """
    if overlap_tokens <= 0 or not chunks:
        return chunks
    spans = token_spans(text)
    starts = [s for s, _ in spans]
    out = [chunks[0]]
    for prev, ch in zip(chunks, chunks[1:]):
        if ch.start < 0:
            out.append(ch)
            continue
        idx = bisect.bisect_left(starts, ch.start)  # first token inside this chunk
        new_idx = max(0, idx - overlap_tokens)
        new_start = max(spans[new_idx][0], prev.start) if spans else ch.start
        md = dict(ch.metadata, overlap_tokens=idx - bisect.bisect_left(starts, new_start))
        out.append(Chunk(text[new_start:ch.end], new_start, ch.end, md))
    return out


LengthFn = Callable[[str], int]
