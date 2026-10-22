"""Strategy 06: agentic chunking (a small model inspects context and places the boundary).

The reel's three steps, per incoming sentence:
  1. INSPECT       - look at the chunk being built and the next sentence
  2. CONTEXTUALIZE - keep a running title/summary of what the chunk is about
  3. BOUNDARY      - decide: does the sentence belong here, or start a new chunk?
"""

from __future__ import annotations

import re
import time
from collections import Counter

from . import events
from .base import Chunk, attach_headings, count_tokens, sentence_spans
from .llm import LLM, ask_json

PROMPT = """You are a boundary inspector for a document chunker.

Current chunk title: {title}
Current chunk (most recent part):
\"\"\"{chunk}\"\"\"

Next sentence:
\"\"\"{sentence}\"\"\"

Does the next sentence continue the SAME topic as the current chunk, or does it
start a NEW topic? Reply with JSON only:
{{"boundary": true or false, "title": "<short title for the chunk the sentence belongs to>"}}
"""

SCHEMA = {
    "type": "object",
    "required": ["boundary", "title"],
    "properties": {"boundary": {"type": "boolean"}, "title": {"type": "string"}},
}

_SHIFT = re.compile(
    r"^(#+\s|however\b|meanwhile\b|next\b|finally\b|in contrast\b|on the other hand\b|"
    r"another\b|turning to\b|moving on\b|separately\b)",
    re.I,
)
_CONTINUE = re.compile(
    r"^(it|its|they|their|this|these|that|those|he|she|so|also|and|but|then|"
    r"because|for example|for instance|in addition|as a result)\b",
    re.I,
)
_EARLY_PRONOUN = re.compile(r"^(?:\S+\s+){0,4}(it|they|them|its|their)\b", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}")
_STOP = {
    "this",
    "that",
    "with",
    "from",
    "they",
    "their",
    "have",
    "will",
    "your",
    "into",
    "when",
    "which",
    "also",
    "than",
    "then",
    "them",
    "these",
    "those",
    "there",
    "about",
    "each",
}


class AgenticChunker:
    """Builds chunks sentence by sentence, letting a model decide every boundary.

    Args:
        llm: `prompt -> text` callable (see chunking.llm). A small/cheap model is fine.
             Without one, an offline heuristic plays the inspector: headings and
             lexical cohesion (shared word stems) decide the boundary.
        max_tokens: hard cap; a chunk is closed before it would exceed this.
        min_sentences: the heuristic won't close a chunk smaller than this.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        max_tokens: int = 300,
        min_sentences: int = 2,
        context_chars: int = 800,
        **_ignored,
    ):
        self.llm = llm
        self.max_tokens = max_tokens
        self.min_sentences = min_sentences
        self.context_chars = context_chars

    def split(self, text: str) -> list[Chunk]:
        spans = attach_headings(text, sentence_spans(text))
        if not spans:
            return []
        sentences = [text[a:b] for a, b in spans]

        chunks: list[Chunk] = []
        mode = "llm" if self.llm else "heuristic"
        events.emit("agentic.start", sentences=len(spans), spans=spans, mode=mode)
        start, title, reason = 0, self._title([sentences[0]]), "start"
        for i in range(1, len(spans) + 1):
            if i < len(spans):
                t0 = time.perf_counter()
                cur_text = text[spans[start][0] : spans[i - 1][1]]
                if count_tokens(cur_text) + count_tokens(sentences[i]) > self.max_tokens:
                    boundary, new_title, why = True, None, "max_tokens"
                elif self.llm:  # step 1: inspect
                    boundary, new_title, why = self._ask_llm(title, cur_text, sentences[i])
                else:
                    boundary, why = self._heuristic(sentences[start:i], sentences[i])
                    new_title = None
                events.emit(
                    "agentic.decision",
                    index=i,
                    start=spans[i][0],
                    end=spans[i][1],
                    boundary=bool(boundary),
                    reason=why or "continues",
                    mode=mode,
                    title=new_title or title,
                    ms=round((time.perf_counter() - t0) * 1000, 2),
                )
            if i == len(spans) or boundary:  # step 3: boundary
                a, b = spans[start][0], spans[i - 1][1]
                chunks.append(
                    Chunk(
                        text[a:b],
                        a,
                        b,
                        {
                            "strategy": "agentic",
                            "title": title,
                            "opened_by": reason,
                            "mode": "llm" if self.llm else "heuristic",
                        },
                    )
                )
                if i == len(spans):
                    break
                start, reason = i, why
                title = new_title or self._title([sentences[i]])
            else:  # step 2: contextualize
                title = new_title or self._title(sentences[start : i + 1])
        return chunks

    # -- decision makers ---------------------------------------------------
    def _ask_llm(self, title, chunk_text, sentence):
        if sentence.lstrip().startswith("#"):
            return True, None, "heading"  # a heading always opens a chunk; skip the model call
        prompt = PROMPT.format(title=title, chunk=chunk_text[-self.context_chars :], sentence=sentence)
        try:
            data = ask_json(self.llm, prompt, SCHEMA)
            boundary = data.get("boundary")
            if isinstance(boundary, str):
                boundary = boundary.strip().lower() in ("true", "yes", "1")
            return bool(boundary), (str(data.get("title") or "").strip() or None), "llm"
        except (ValueError, AttributeError):
            return False, None, ""  # unparseable answer -> keep going, don't break the chunk

    def _heuristic(self, chunk_sentences, sentence):
        """Offline stand-in for the model."""
        s = sentence.lstrip()
        if s.startswith("#"):
            return True, "heading"
        if _CONTINUE.match(s) or _EARLY_PRONOUN.match(s):
            return False, ""  # "It ...", "Teams use it ..." -> continues
        shared = _stems(sentence) & _stems(" ".join(chunk_sentences))
        if len(chunk_sentences) < self.min_sentences:
            return False, ""
        if not shared:
            return True, "no shared terms"
        if _SHIFT.match(s) and len(shared) <= 1:
            return True, "topic-shift cue"
        return False, ""

    @staticmethod
    def _title(sentences: list[str]) -> str:
        first = sentences[0].strip()
        if first.startswith("#"):
            return first.splitlines()[0].lstrip("# ").strip()
        words = [w.lower() for s in sentences for w in _WORD.findall(s) if w.lower() not in _STOP]
        top = [w for w, _ in Counter(words).most_common(3)]
        return " / ".join(top) if top else first[:40]


def _stems(text: str) -> set:
    """Crude stemming: first 5 letters of each content word ("evicts"/"eviction" -> "evict")."""
    return {w.lower()[:5] for w in _WORD.findall(text) if w.lower() not in _STOP}
