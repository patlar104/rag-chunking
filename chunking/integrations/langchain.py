"""LangChain integration.

    pip install langchain-core langchain-text-splitters

Use any of the six strategies as a LangChain text splitter:

    from chunking.integrations.langchain import ChunkingTextSplitter
    splitter = ChunkingTextSplitter("document", max_tokens=300)
    docs = splitter.split_documents(loader.load())        # metadata: headings, start_index, ...

Plug LangChain models into the chunkers (any Embeddings / chat model works):

    from langchain_ollama import OllamaEmbeddings, ChatOllama
    splitter = ChunkingTextSplitter("semantic", embeddings=OllamaEmbeddings(model="nomic-embed-text"))
    splitter = ChunkingTextSplitter("agentic", llm=ChatOllama(model="llama3.2:3b"))

Or use this package's embedders (Ollama, sentence-transformers, cache) inside LangChain:

    from chunking.integrations.langchain import ChunkingEmbeddings
    store = InMemoryVectorStore(ChunkingEmbeddings("ollama:nomic-embed-text", cache=True))
"""
from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import TextSplitter
except ImportError as e:  # pragma: no cover
    raise ImportError("LangChain integration needs: pip install langchain-core langchain-text-splitters") from e

from .. import events
from ..embeddings import embed_query, get_embedder
from ._common import build_chunker, chunk_metadata, model_label, reply_text, resolve_embedder, resolve_llm, run_chunker

__all__ = ["ChunkingTextSplitter", "ChunkingEmbeddings", "LangChainEmbedder", "LangChainLLM"]


# -- LangChain models -> chunking backends ------------------------------------------------

class LangChainEmbedder:
    """Wrap a LangChain `Embeddings` so the semantic chunker can use it."""

    def __init__(self, embeddings: Embeddings):
        self.lc = embeddings
        self.name = model_label(embeddings, "langchain")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        with events.timed("embed.call", backend=self.name, n=len(texts)):
            return self.lc.embed_documents(list(texts))

    def query(self, text: str) -> list[float]:
        with events.timed("embed.call", backend=self.name, n=1):
            return self.lc.embed_query(text)


class LangChainLLM:
    """Wrap a LangChain chat model / LLM (anything with `.invoke`) as a `str -> str` callable."""

    def __init__(self, model: Any):
        self.lc = model
        self.name = model_label(model, "langchain")
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        with events.timed("llm.call", backend=self.name, json=False, prompt_chars=len(prompt)):
            return reply_text(self.lc.invoke(prompt))


def _as_embedder(embeddings):
    embeddings = resolve_embedder(embeddings)
    if isinstance(embeddings, ChunkingEmbeddings):
        return embeddings.embedder          # already one of ours: skip the round trip
    if isinstance(embeddings, Embeddings) or (
            hasattr(embeddings, "embed_documents") and hasattr(embeddings, "embed_query")):
        return LangChainEmbedder(embeddings)
    return embeddings


def _as_llm(llm):
    llm = resolve_llm(llm)
    if llm is not None and hasattr(llm, "invoke") and not hasattr(llm, "complete_json"):
        return LangChainLLM(llm)
    return llm


# -- chunkers -> LangChain text splitter -----------------------------------------------------

class ChunkingTextSplitter(TextSplitter):
    """A LangChain `TextSplitter` backed by any strategy in this package.

    Args:
        strategy: fixed | recursive | document | semantic | propositional | agentic
        chunker: pass a ready-made chunker instead of a strategy name.
        embeddings: for "semantic": a LangChain `Embeddings`, one of our embedders, or a
            spec string such as "ollama:nomic-embed-text".
        llm: for "agentic"/"propositional": a LangChain chat model, one of our LLMs, or a
            spec string such as "ollama:llama3.2:3b".
        overlap_tokens: sliding-window overlap added after chunking (offset-preserving strategies).
        flatten_metadata: make metadata vector-store safe (only str/int/float/bool).
        **chunker_kwargs: forwarded to the chunker, e.g. chunk_size=200 or max_tokens=300.
    """

    def __init__(self, strategy: str = "recursive", *, chunker=None, embeddings=None, llm=None,
                 overlap_tokens: int = 0, add_start_index: bool = True, flatten_metadata: bool = True,
                 **chunker_kwargs: Any):
        super().__init__(chunk_size=10 ** 9, chunk_overlap=0, add_start_index=add_start_index,
                         strip_whitespace=False)
        self.strategy = strategy
        self.chunker = build_chunker(strategy, chunker, _as_embedder(embeddings), _as_llm(llm), **chunker_kwargs)
        self.overlap_tokens = overlap_tokens
        self.flatten_metadata = flatten_metadata

    def split_chunks(self, text: str):
        """Return this package's `Chunk` objects (text + offsets + metadata)."""
        return run_chunker(self.chunker, text, self.overlap_tokens)

    def split_text(self, text: str) -> list[str]:
        return [c.text for c in self.split_chunks(text)]

    def create_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        metadatas = metadatas or [{}] * len(texts)
        docs = []
        for text, base in zip(texts, metadatas):
            for c in self.split_chunks(text):
                md = copy.deepcopy(base)
                cm = chunk_metadata(c, self.flatten_metadata)
                if not self._add_start_index:
                    cm.pop("start_index", None)
                    cm.pop("end_index", None)
                md.update(cm)
                md.setdefault("strategy", self.strategy)
                docs.append(Document(page_content=c.text, metadata=md))
        return docs

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        documents = list(documents)
        return self.create_documents([d.page_content for d in documents], [d.metadata for d in documents])


# -- chunking embedders -> LangChain Embeddings ------------------------------------------------

class ChunkingEmbeddings(Embeddings):
    """Expose this package's embedders (Ollama with correct prefixes, sentence-transformers,
    OpenAI, SQLite cache) as a LangChain `Embeddings`."""

    def __init__(self, embedder: Any = "auto", cache: bool = False, **kwargs: Any):
        self.embedder = get_embedder(embedder, cache=cache, **kwargs) if isinstance(embedder, str) else embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return embed_query(self.embedder, text)
