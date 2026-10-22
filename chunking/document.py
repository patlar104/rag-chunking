"""Strategy 03: document-based chunking (preserve Markdown headers, tables, code)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import Chunk, LengthFn, count_tokens, strip_span
from .recursive import RecursiveChunker

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class Block:
    kind: str  # heading | code | table | text
    start: int
    end: int
    level: int = 0
    title: str = ""


class MarkdownChunker:
    """Chunk along the document's own structure.

    * Every heading starts a new section; each chunk carries its heading path
      (e.g. ["Installation", "Requirements"]) as metadata.
    * Code blocks and tables are atomic: they are never split, even if they
      are larger than `max_tokens` (a half table is worse than a big chunk).
    * If a section is still too large, it is split between blocks, and only
      long prose blocks fall back to recursive splitting.
    """

    def __init__(self, max_tokens: int = 300, include_heading_path: bool = True, length_fn: LengthFn = count_tokens):
        self.max_tokens = max_tokens
        self.include_heading_path = include_heading_path
        self.length = length_fn
        self._prose = RecursiveChunker(max_tokens, length_fn=length_fn)

    # -- parsing ---------------------------------------------------------
    def parse(self, text: str) -> list[Block]:
        blocks: list[Block] = []
        lines = text.splitlines(keepends=True)
        offsets, pos = [], 0
        for ln in lines:
            offsets.append(pos)
            pos += len(ln)
        i = 0
        while i < len(lines):
            line, start = lines[i], offsets[i]
            if _FENCE.match(line):
                fence = _FENCE.match(line).group(1)
                j = i + 1
                while j < len(lines) and not lines[j].lstrip().startswith(fence):
                    j += 1
                j = min(j + 1, len(lines))
                blocks.append(Block("code", start, offsets[j - 1] + len(lines[j - 1])))
                i = j
            elif _HEADING.match(line):
                m = _HEADING.match(line)
                blocks.append(
                    Block("heading", start, start + len(line.rstrip("\n")), level=len(m.group(1)), title=m.group(2))
                )
                i += 1
            elif _TABLE.match(line):
                j = i
                while j < len(lines) and _TABLE.match(lines[j]):
                    j += 1
                blocks.append(Block("table", start, offsets[j - 1] + len(lines[j - 1])))
                i = j
            elif line.strip() == "":
                i += 1
            else:
                j = i
                while (
                    j < len(lines)
                    and lines[j].strip()
                    and not _HEADING.match(lines[j])
                    and not _FENCE.match(lines[j])
                    and not _TABLE.match(lines[j])
                ):
                    j += 1
                blocks.append(Block("text", start, offsets[j - 1] + len(lines[j - 1])))
                i = j
        return blocks

    # -- chunking --------------------------------------------------------
    def split(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        path: list[tuple] = []  # stack of (level, title)
        group: list[Block] = []

        def flush():
            if not group:
                return
            heading_path = [t for _, t in path]
            for piece in self._pack(text, group):
                a, b, kinds = piece
                a, b = strip_span(text, a, b)
                body = text[a:b]
                md = {"strategy": "document", "headings": heading_path, "blocks": kinds}
                if self.include_heading_path and heading_path:
                    # Re-attach the section context so the chunk stands alone.
                    ctx = heading_path[:-1] if body.startswith("#") else heading_path
                    if ctx:
                        body = " > ".join(ctx) + "\n" + body
                chunks.append(Chunk(body, a, b, md))
            group.clear()

        for blk in self.parse(text):
            if blk.kind == "heading":
                flush()
                while path and path[-1][0] >= blk.level:
                    path.pop()
                path.append((blk.level, blk.title))
            group.append(blk)
        flush()
        return chunks

    def _pack(self, text: str, blocks: list[Block]):
        """Greedily pack whole blocks into pieces of <= max_tokens."""
        pieces, cur = [], []

        def emit():
            if cur:
                pieces.append((cur[0].start, cur[-1].end, [b.kind for b in cur]))
                cur.clear()

        for b in blocks:
            size = self.length(text[b.start : b.end])
            if b.kind == "text" and size > self.max_tokens:
                head = cur.pop() if cur and cur[-1].kind == "heading" else None
                emit()
                for k, c in enumerate(self._prose.split(text[b.start : b.end])):
                    glue = head is not None and k == 0
                    pieces.append(
                        (
                            head.start if glue else b.start + c.start,
                            b.start + c.end,
                            ["heading", "text"] if glue else ["text"],
                        )
                    )
                continue
            if cur and self.length(text[cur[0].start : b.end]) > self.max_tokens:
                # keep a heading glued to whatever follows it
                carry = [cur.pop()] if cur[-1].kind == "heading" else []
                emit()
                cur.extend(carry)
            cur.append(b)
        emit()
        return pieces
