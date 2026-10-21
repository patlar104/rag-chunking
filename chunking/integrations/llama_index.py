"""LlamaIndex integration.

    pip install llama-index-core

Use any of the six strategies as a LlamaIndex node parser:

    from chunking.integrations.llama_index import ChunkingNodeParser
    parser = ChunkingNodeParser("document", max_tokens=300)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)        # or: VectorStoreIndex.from_documents(docs, transformations=[parser])

Plug LlamaIndex models into the chunkers:

    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    ChunkingNodeParser("semantic", embed_model=OllamaEmbedding(model_name="nomic-embed-text"))
    ChunkingNodeParser("agentic", llm=Ollama(model="llama3.2:3b", request_timeout=120))

Or use this package's embedders inside LlamaIndex:

    from chunking.integrations.llama_index import ChunkingEmbedding
    Settings.embed_model = ChunkingEmbedding("ollama:nomic-embed-text", cache=True)
"""
from __future__ import annotations

from typing import Any, List, Sequence

try:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.bridge.pydantic import Field, PrivateAttr
    from llama_index.core.node_parser import NodeParser
    from llama_index.core.node_parser.node_utils import build_nodes_from_splits
    from llama_index.core.schema import BaseNode
    from llama_index.core.utils import get_tqdm_iterable
except ImportError as e:  # pragma: no cover
    raise ImportError("LlamaIndex integration needs: pip install llama-index-core") from e

from .. import events
from ..embeddings import embed_query, get_embedder
from ._common import (
    NON_EMBED_KEYS,
    build_chunker,
    chunk_metadata,
    model_label,
    reply_text,
    resolve_embedder,
    resolve_llm,
    run_chunker,
)

__all__ = ["ChunkingNodeParser", "ChunkingEmbedding", "LlamaIndexEmbedder", "LlamaIndexLLM"]


# -- LlamaIndex models -> chunking backends -----------------------------------------------------

class LlamaIndexEmbedder:
    """Wrap a LlamaIndex embedding model so the semantic chunker can use it."""

    def __init__(self, embed_model: BaseEmbedding):
        self.li = embed_model
        self.name = model_label(embed_model, "llamaindex")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        with events.timed("embed.call", backend=self.name, n=len(texts)):
            return self.li.get_text_embedding_batch(list(texts))

    def query(self, text: str) -> List[float]:
        with events.timed("embed.call", backend=self.name, n=1):
            return self.li.get_query_embedding(text)


class LlamaIndexLLM:
    """Wrap a LlamaIndex LLM (anything with `.complete`) as a `str -> str` callable."""

    def __init__(self, llm: Any):
        self.li = llm
        model = getattr(llm, "model", None) or getattr(getattr(llm, "metadata", None), "model_name", None)
        self.name = f"llamaindex:{model or type(llm).__name__}"
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        with events.timed("llm.call", backend=self.name, json=False, prompt_chars=len(prompt)):
            return reply_text(self.li.complete(prompt))


def _as_embedder(embed_model):
    embed_model = resolve_embedder(embed_model)
    if isinstance(embed_model, ChunkingEmbedding):
        return embed_model._embedder
    if isinstance(embed_model, BaseEmbedding):
        return LlamaIndexEmbedder(embed_model)
    return embed_model


def _as_llm(llm):
    llm = resolve_llm(llm)
    if llm is not None and hasattr(llm, "complete") and not hasattr(llm, "complete_json"):
        return LlamaIndexLLM(llm)
    return llm


# -- chunkers -> LlamaIndex node parser ------------------------------------------------------------

class ChunkingNodeParser(NodeParser):
    """A LlamaIndex `NodeParser` backed by any strategy in this package.

    Each node gets `start_char_idx`/`end_char_idx` from the chunker's real offsets,
    the chunk metadata (heading path, title, cut reason, ...), and SOURCE/PREV/NEXT
    relationships. Bookkeeping keys are excluded from the embedding text, and
    propositional chunks keep their `source_passage` for the LLM but not for embedding.
    """

    strategy: str = Field(default="recursive", description="chunking strategy name")
    overlap_tokens: int = Field(default=0, description="sliding-window overlap in tokens")
    flatten_metadata: bool = Field(default=True, description="make metadata vector-store safe")
    _chunker: Any = PrivateAttr()

    def __init__(self, strategy: str = "recursive", *, chunker=None, embed_model=None, llm=None,
                 overlap_tokens: int = 0, flatten_metadata: bool = True,
                 include_metadata: bool = True, include_prev_next_rel: bool = True,
                 **chunker_kwargs: Any):
        super().__init__(strategy=strategy, overlap_tokens=overlap_tokens, flatten_metadata=flatten_metadata,
                         include_metadata=include_metadata, include_prev_next_rel=include_prev_next_rel)
        self._chunker = build_chunker(strategy, chunker, _as_embedder(embed_model), _as_llm(llm), **chunker_kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ChunkingNodeParser"

    @property
    def chunker(self):
        return self._chunker

    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        out: List[BaseNode] = []
        for node in get_tqdm_iterable(nodes, show_progress, "Chunking"):
            text = node.get_content()
            chunks = run_chunker(self._chunker, text, self.overlap_tokens)
            built = build_nodes_from_splits([c.text for c in chunks], node, id_func=self.id_func)
            for n, c in zip(built, chunks):
                md = chunk_metadata(c, self.flatten_metadata)
                md.setdefault("strategy", self.strategy)
                n.metadata.update(md)
                n.excluded_embed_metadata_keys = list(n.excluded_embed_metadata_keys) + \
                    [k for k in md if k in NON_EMBED_KEYS]
                n.excluded_llm_metadata_keys = list(n.excluded_llm_metadata_keys) + \
                    [k for k in md if k in NON_EMBED_KEYS and k != "source_passage"]
                if c.start >= 0:
                    n.start_char_idx, n.end_char_idx = c.start, c.end
            out.extend(built)
        return out


# -- chunking embedders -> LlamaIndex embed model ------------------------------------------------------

class ChunkingEmbedding(BaseEmbedding):
    """Expose this package's embedders (Ollama with correct prefixes, sentence-transformers,
    OpenAI, SQLite cache) as a LlamaIndex embedding model."""

    _embedder: Any = PrivateAttr()

    def __init__(self, embedder: Any = "auto", cache: bool = False, embed_batch_size: int = 32, **kwargs: Any):
        emb = get_embedder(embedder, cache=cache, **kwargs) if isinstance(embedder, str) else embedder
        super().__init__(model_name=getattr(emb, "name", type(emb).__name__), embed_batch_size=embed_batch_size)
        self._embedder = emb

    @classmethod
    def class_name(cls) -> str:
        return "ChunkingEmbedding"

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embedder([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embedder(list(texts))

    def _get_query_embedding(self, query: str) -> List[float]:
        return embed_query(self._embedder, query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
