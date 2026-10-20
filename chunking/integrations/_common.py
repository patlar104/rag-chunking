"""Shared helpers for the framework integrations."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .. import get_chunker, with_overlap
from ..base import Chunk

# Keys that are useful to store but shouldn't be embedded (they would dilute the vector).
NON_EMBED_KEYS = ("source_passage", "start_index", "end_index", "strategy", "split_level", "blocks",
                  "cut_reason", "cut_distance", "threshold", "sentences", "opened_by", "mode",
                  "first_token", "overlap_tokens")


def flatten(md: Dict[str, Any]) -> Dict[str, Any]:
    """Vector stores (Chroma, Pinecone, ...) only accept str/int/float/bool values."""
    out: Dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
            out[k] = " > ".join(v) if k == "headings" else ", ".join(v)
        else:
            out[k] = json.dumps(v, default=str)
    return out


def chunk_metadata(chunk: Chunk, flat: bool = True) -> Dict[str, Any]:
    md = dict(chunk.metadata)
    if chunk.start >= 0:
        md["start_index"], md["end_index"] = chunk.start, chunk.end
    return flatten(md) if flat else md


def build_chunker(strategy: str, chunker=None, embed=None, llm=None, **kwargs):
    if chunker is not None:
        return chunker
    if strategy == "semantic" and embed is not None:
        kwargs["embed"] = embed
    if strategy in ("agentic", "propositional") and llm is not None:
        kwargs["llm"] = llm
    return get_chunker(strategy, **kwargs)


def run_chunker(chunker, text: str, overlap_tokens: int = 0):
    chunks = chunker.split(text)
    if overlap_tokens and chunks and all(c.start >= 0 and text[c.start:c.end] == c.text for c in chunks):
        chunks = with_overlap(text, chunks, overlap_tokens)
    return chunks


def reply_text(resp: Any) -> str:
    """Normalise LangChain / LlamaIndex responses to a string."""
    for attr in ("content", "text"):
        val = getattr(resp, attr, None)
        if val is not None:
            if isinstance(val, list):  # content blocks
                return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in val)
            return str(val)
    message = getattr(resp, "message", None)
    if message is not None:
        return reply_text(message)
    return str(resp)


def model_label(obj: Any, prefix: str) -> str:
    for attr in ("model", "model_name", "model_id"):
        val = getattr(obj, attr, None)
        if isinstance(val, str) and val:
            return f"{prefix}:{val}"
    return f"{prefix}:{type(obj).__name__}"


def resolve_embedder(embed: Optional[Any]):
    """Accept a spec string ('ollama:nomic-embed-text') or an embedder object."""
    if isinstance(embed, str):
        from ..embeddings import get_embedder
        return get_embedder(embed)
    return embed


def resolve_llm(llm: Optional[Any]):
    if isinstance(llm, str):
        from ..llm import get_llm
        return get_llm(llm)
    return llm
