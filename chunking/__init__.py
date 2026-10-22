"""Six production chunking strategies for RAG, plus the sliding-window overlap fix.

from chunking import get_chunker, get_embedder, get_llm, with_overlap
chunks = get_chunker("recursive", chunk_size=120).split(text)
chunks = get_chunker("semantic", embed=get_embedder("ollama:nomic-embed-text")).split(text)
chunks = get_chunker("agentic", llm=get_llm("ollama:llama3.2:3b")).split(text)
"""

import sys

if sys.version_info < (3, 13):  # noqa: UP036 -- fail fast with a clear message instead of a confusing SyntaxError/TypeError
    raise ImportError(
        f"rag-chunking needs Python 3.13 or newer (3.15 recommended); this is Python {sys.version.split()[0]}. "
        "See the Setup section of the README."
    )

from .agentic import AgenticChunker
from .base import Chunk, count_tokens, sentence_spans, with_overlap
from .document import MarkdownChunker
from .embeddings import get_embedder
from .fixed import FixedSizeChunker
from .llm import get_llm
from .propositional import PropositionalChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .stats import chunk_stats

STRATEGIES = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "document": MarkdownChunker,
    "semantic": SemanticChunker,
    "propositional": PropositionalChunker,
    "agentic": AgenticChunker,
}


def get_chunker(name: str, **kwargs):
    try:
        return STRATEGIES[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; pick one of {sorted(STRATEGIES)}") from None


__all__ = [
    "Chunk",
    "STRATEGIES",
    "get_chunker",
    "get_embedder",
    "get_llm",
    "with_overlap",
    "chunk_stats",
    "count_tokens",
    "sentence_spans",
    *[c.__name__ for c in STRATEGIES.values()],
]
