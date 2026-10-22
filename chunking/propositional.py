"""Strategy 05: propositional chunking (decompose complex text into atomic facts)."""

from __future__ import annotations

import re

from . import events
from .base import Chunk, sentence_spans
from .llm import LLM, ask_json

PROMPT = """Decompose the passage into atomic propositions.
Rules:
1. Each proposition states exactly ONE fact and must make sense on its own.
2. Replace pronouns (it, they, this, he, she) with the full entity they refer to.
3. Keep the original meaning; do not add outside knowledge.
4. Skip headings, tables and code; only extract facts from prose.
5. Reply with JSON only: {{"propositions": ["...", "..."]}}

Passage:
{passage}
"""

SCHEMA = {
    "type": "object",
    "required": ["propositions"],
    "properties": {"propositions": {"type": "array", "items": {"type": "string"}}},
}

_PRONOUN = re.compile(r"^(It|They|This|These|Its|Their)\b")
_INNER_PRONOUN = re.compile(r"^(it|they)\b")
_AUX = {
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "can",
    "could",
    "will",
    "would",
    "must",
    "should",
    "may",
    "might",
    "does",
    "do",
    "did",
}
_DET = {"the", "a", "an", "this", "that", "these", "those", "each", "every", "many", "some"}
# split "X, and it Y" / "X; Y" / "X, while the Z" into separate clauses
_SPLIT = re.compile(
    r";\s+|,\s+(?:and|but|while|whereas)\s+(?=(?:it|they|the|a|an|[A-Z])\b)|"
    r"\s+(?:but|while|whereas)\s+(?=(?:it|they|the)\b)"
)
_SUBORDINATE = {
    "when",
    "if",
    "while",
    "after",
    "before",
    "because",
    "although",
    "since",
    "once",
    "however,",
    "unless",
    "until",
    "as",
}


def _verbish(word: str) -> bool:
    w = word.lower()
    return w in _AUX or (
        w.isalpha()
        and len(w) > 3
        and w.endswith(("s", "ed"))
        and not w.endswith("ss")
        and w not in {"this", "its", "thus", "always", "towards"}
    )


def _subject(sentence: str) -> str | None:
    """Leading noun phrase, e.g. 'The append-only file loses ...' -> 'The append-only file'.
    Conservative: returns None unless the phrase looks like a real subject."""
    words = sentence.split()
    if not words or _PRONOUN.match(words[0]) or words[0].lower() in _SUBORDINATE:
        return None
    first = 2 if words[0].lower() in _DET else 1
    adverb_at = None
    for i in range(first, min(len(words), 7)):
        w = words[i].strip(",")
        if w.endswith("ly") and w.lower() not in _AUX:
            adverb_at = adverb_at or i  # 'policy only evicts' -> stop before 'only'
            continue
        if w.lower() in _AUX or (_verbish(w) and words[i - 1].lower() not in _DET):
            return " ".join(words[: adverb_at or i]).strip(",")
        if words[i - 1].endswith(","):
            return None
    return None


def _is_structured(s: str) -> bool:
    return "```" in s or any(ln.lstrip().startswith("|") for ln in s.splitlines())


class PropositionalChunker:
    """Each chunk is one self-contained fact, which makes retrieval very precise.

    With an `llm`, a model rewrites each passage into propositions (the
    approach from the "Dense X Retrieval" paper). Without one, a rule-based
    fallback splits compound sentences and resolves leading pronouns, so you
    can still see how it works offline.
    """

    def __init__(self, llm: LLM | None = None, passage_sentences: int = 4):
        self.llm = llm
        self.passage_sentences = passage_sentences

    def split(self, text: str) -> list[Chunk]:
        # Code blocks and tables aren't prose facts: skip them (and save model calls).
        spans = [sp for sp in sentence_spans(text) if not _is_structured(text[sp[0] : sp[1]])]
        chunks: list[Chunk] = []
        subject = None
        for i in range(0, len(spans), self.passage_sentences):
            group = spans[i : i + self.passage_sentences]
            a, b = group[0][0], group[-1][1]
            passage = "\n".join(ln for ln in text[a:b].splitlines() if not ln.lstrip().startswith("#")).strip()
            if not passage:
                continue
            props = None
            if self.llm:
                try:
                    props, mode = self._ask(passage), "llm"
                except ValueError as e:
                    props = None  # model returned junk twice; use the offline rules for this passage
                    events.emit("fallback", strategy="propositional", reason=str(e)[:200])
            if props is None:
                props, subject = self._rule_based([text[x:y] for x, y in group], subject)
                mode = "rule-based" if not self.llm else "rule-based (llm reply unusable)"
            events.emit("propositional.passage", start=a, end=b, propositions=props, mode=mode)
            for p in props:
                chunks.append(Chunk(p, a, b, {"strategy": "propositional", "mode": mode, "source_passage": passage}))
        return chunks

    def _ask(self, passage: str) -> list[str]:
        data = ask_json(self.llm, PROMPT.format(passage=passage), SCHEMA)
        if isinstance(data, dict):
            data = data.get("propositions", next((v for v in data.values() if isinstance(v, list)), []))
        if not isinstance(data, list):
            raise ValueError("expected a list of propositions")
        return [str(p).strip() for p in data if str(p).strip()]

    # -- offline fallback --------------------------------------------------
    @staticmethod
    def _rule_based(sentences: list[str], subject: str | None):
        """Split compound sentences into clauses and replace leading pronouns.
        Approximate by design; use an LLM for real propositional chunking."""
        props = []
        for s in sentences:
            if "```" in s or any(ln.lstrip().startswith("|") for ln in s.splitlines()):
                continue  # code and tables aren't prose facts
            lines = [ln for ln in s.splitlines() if not ln.lstrip().startswith("#")]
            s = " ".join(" ".join(lines).split())
            if not s:
                continue
            own = _subject(s)
            subject = own or subject
            for k, part in enumerate(_SPLIT.split(s)):
                part = part.strip(" ,;")
                if len(part.split()) < 3:
                    continue
                if subject and (_PRONOUN.match(part) or (k > 0 and _INNER_PRONOUN.match(part))):
                    poss = part.split()[0].lower() in ("its", "their")
                    part = re.sub(r"^\w+", subject + ("'s" if poss else ""), part, count=1)
                elif k > 0 and own and _verbish(part.split()[0]):
                    part = f"{own} {part}"  # "... and supports X" -> "Redis supports X"
                part = part[0].upper() + part[1:]
                props.append(part if part.endswith((".", "!", "?")) else part + ".")
        return props, subject
