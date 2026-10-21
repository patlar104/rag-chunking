"""Embedding backends.

An embedder is any `Callable[[list[str]], list[list[float]]]` (document texts ->
vectors). The classes here also expose `.query(text)` for asymmetric models that
want a different prefix for search queries (nomic-embed-text, e5, bge, ...).

    get_embedder("hashing")                       # offline toy, no downloads
    get_embedder("ollama:nomic-embed-text")       # local, via Ollama
    get_embedder("st:all-MiniLM-L6-v2")           # local, via sentence-transformers
    get_embedder("openai:text-embedding-3-small") # API
    get_embedder("auto")                          # best one available on this machine
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from . import events, ollama

Embedder = Callable[[List[str]], List[List[float]]]

# Instruction prefixes for popular asymmetric models: (document prefix, query prefix).
# Getting these wrong silently costs retrieval quality, so we apply them for you.
KNOWN_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
    "mixedbread-ai/mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
    "e5": ("passage: ", "query: "),
    "bge": ("", "Represent this sentence for searching relevant passages: "),
    "snowflake-arctic-embed": ("", "Represent this sentence for searching relevant passages: "),
}


def _prefixes_for(model: str) -> tuple:
    name = model.lower()
    for key, pair in KNOWN_PREFIXES.items():
        if key in name:
            return pair
    return ("", "")


class BaseEmbedder:
    """Shared behaviour: batching, prefixes, normalisation, `.query()`."""

    name = "base"
    batch_size = 64

    def __init__(self, doc_prefix: Optional[str] = None, query_prefix: Optional[str] = None,
                 model: str = ""):
        auto_doc, auto_query = _prefixes_for(model)
        self.doc_prefix = auto_doc if doc_prefix is None else doc_prefix
        self.query_prefix = auto_query if query_prefix is None else query_prefix

    def _embed(self, texts: List[str]) -> List[List[float]]:  # implement in subclasses
        raise NotImplementedError

    def _run(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            with events.timed("embed.call", backend=self.name, n=len(batch)):
                out.extend(_normalize(v) for v in self._embed(batch))
        return out

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self._run([self.doc_prefix + t for t in texts]) if texts else []

    def query(self, text: str) -> List[float]:
        return self._run([self.query_prefix + text])[0]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


# -- offline -----------------------------------------------------------------

_STOP = set("""a an and are as at be been but by can could did do does for from had has have he her
his how i if in into is it its just may me more most my no not of on or our she so some such than
that the their them then there these they this those to too very was we were what when where which
while who why will with would you your""".split())


def _stem(w: str) -> str:
    for suf in ("ingly", "edly", "ing", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


class HashingEmbedder(BaseEmbedder):
    """Bag-of-words hashed into a fixed vector (+ character trigrams).
    Dependency-free and instant, but purely lexical: it can't tell that
    "borrow checker" is about Rust. Use a real model for anything serious."""

    name = "hashing"

    def __init__(self, dim: int = 512):
        super().__init__("", "")
        self.dim = dim

    def _embed(self, texts):
        vecs = []
        for text in texts:
            v = [0.0] * self.dim
            words = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]
            for f in words + [w[i:i + 3] for w in words for i in range(max(1, len(w) - 2))]:
                v[int(hashlib.md5(f.encode()).hexdigest(), 16) % self.dim] += 1.0
            vecs.append(v)
        return vecs


# -- Ollama --------------------------------------------------------------------

class OllamaEmbedder(BaseEmbedder):
    """Local embeddings from an Ollama server.

        ollama pull nomic-embed-text      # 274 MB, 768-d, good default
        ollama pull mxbai-embed-large     # 670 MB, 1024-d, stronger
        ollama pull all-minilm            #  46 MB, 384-d, fastest

    Uses the batched `/api/embed` endpoint (Ollama >= 0.3.4) and falls back to
    the older one-text-at-a-time `/api/embeddings` endpoint automatically.
    """

    batch_size = 32

    def __init__(self, model: str = "nomic-embed-text", host: Optional[str] = None,
                 doc_prefix: Optional[str] = None, query_prefix: Optional[str] = None,
                 truncate: bool = True, keep_alive: str = "5m", timeout: float = 120):
        super().__init__(doc_prefix, query_prefix, model)
        self.model, self.name = model, f"ollama:{model}"
        self.client = ollama.Client(host, timeout=timeout)
        self.truncate, self.keep_alive = truncate, keep_alive
        self._legacy = False

    def _embed(self, texts):
        if not self._legacy:
            try:
                r = self.client.post("/api/embed", {"model": self.model, "input": texts,
                                                    "truncate": self.truncate, "keep_alive": self.keep_alive})
                return r["embeddings"]
            except ollama.EndpointMissing:
                self._legacy = True
        return [self.client.post("/api/embeddings", {"model": self.model, "prompt": t,
                                                     "keep_alive": self.keep_alive})["embedding"]
                for t in texts]


# -- sentence-transformers -----------------------------------------------------

class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings with `pip install sentence-transformers`.

    Good models to try: all-MiniLM-L6-v2 (fast, 384-d), BAAI/bge-small-en-v1.5,
    intfloat/e5-small-v2, nomic-ai/nomic-embed-text-v1.5 (needs trust_remote_code).
    The model is downloaded from Hugging Face on first use, then cached.
    """

    batch_size = 64

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: Optional[str] = None,
                 doc_prefix: Optional[str] = None, query_prefix: Optional[str] = None,
                 trust_remote_code: bool = False):
        super().__init__(doc_prefix, query_prefix, model)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError("sentence-transformers is not installed. Run:\n"
                              "  pip install sentence-transformers") from e
        self.model_name, self.name = model, f"st:{model}"
        self.model = SentenceTransformer(model, device=device, trust_remote_code=trust_remote_code)

    def _embed(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()


# -- OpenAI --------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """`pip install openai` and set OPENAI_API_KEY. Also works with any
    OpenAI-compatible server via `base_url` (LM Studio, vLLM, Ollama's /v1)."""

    batch_size = 256

    def __init__(self, model: str = "text-embedding-3-small", base_url: Optional[str] = None,
                 api_key: Optional[str] = None, dimensions: Optional[int] = None):
        super().__init__("", "", model)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("openai is not installed. Run:\n  pip install openai") from e
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.name, self.dimensions = model, f"openai:{model}", dimensions

    def _embed(self, texts):
        kw = {"dimensions": self.dimensions} if self.dimensions else {}
        data = self.client.embeddings.create(model=self.model, input=texts, **kw).data
        return [d.embedding for d in sorted(data, key=lambda d: d.index)]


# -- cache -----------------------------------------------------------------------

class CachedEmbedder:
    """Wrap any embedder with a SQLite cache keyed by (model, text).

    Semantic chunking and repeated demo runs embed the same sentences over and
    over; with a cache, only new text hits the model.
    """

    def __init__(self, inner, path: Optional[str] = None):
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        default = Path(os.environ.get("CHUNKING_CACHE_DIR", Path.home() / ".cache" / "rag-chunking"))
        self.path = Path(path) if path else default / "embeddings.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec TEXT)")
        self.hits = self.misses = 0

    def _key(self, kind: str, text: str) -> str:
        return hashlib.sha256(f"{self.name}\x00{kind}\x00{text}".encode()).hexdigest()

    def _lookup(self, kind: str, texts: List[str], compute):
        keys = [self._key(kind, t) for t in texts]
        found = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})"
            found.update({k: json.loads(v) for k, v in self.db.execute(q, part)})
        todo = [i for i, k in enumerate(keys) if k not in found]
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        events.emit("cache.lookup", backend=self.name, hits=len(texts) - len(todo), misses=len(todo))
        if todo:
            fresh = compute([texts[i] for i in todo])
            rows = [(keys[i], json.dumps(v)) for i, v in zip(todo, fresh)]
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self.db.commit()
            found.update(dict((k, json.loads(v)) for k, v in rows))
        return [found[k] for k in keys]

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self._lookup("doc", texts, self.inner)

    def query(self, text: str) -> List[float]:
        q = getattr(self.inner, "query", None)
        return self._lookup("query", [text], lambda ts: [q(ts[0]) if q else self.inner(ts)[0]])[0]

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"<CachedEmbedder {self.name} hits={self.hits} misses={self.misses}>"


# -- factory -----------------------------------------------------------------------

def get_embedder(spec: str = "hashing", cache: bool = False, **kwargs):
    """Build an embedder from a short spec: 'hashing', 'ollama[:model]',
    'st[:model]', 'openai[:model]', 'deepinfra[:model]', or 'auto'."""
    kind, _, model = spec.partition(":")
    kind = kind.lower()
    if kind == "auto":
        emb = _auto(**kwargs)
    elif kind == "hashing":
        emb = HashingEmbedder()
    elif kind == "ollama":
        emb = OllamaEmbedder(model or "nomic-embed-text", **kwargs)
    elif kind in ("st", "sentence-transformers", "hf"):
        emb = SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2", **kwargs)
    elif kind == "openai":
        emb = OpenAIEmbedder(model or "text-embedding-3-small", **kwargs)
    elif kind == "deepinfra":
        from .deepinfra import DEFAULT_EMBED, DeepInfraEmbedder
        emb = DeepInfraEmbedder(model or DEFAULT_EMBED, **kwargs)
    else:
        raise ValueError(f"unknown embedder {spec!r}; use hashing, ollama[:model], st[:model], openai[:model], "
                         "deepinfra[:model] or auto")
    return CachedEmbedder(emb) if cache and kind != "hashing" else emb


_OLLAMA_PREFERENCE = ("nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed", "bge-m3",
                      "bge", "embeddinggemma", "qwen3-embedding", "granite-embedding", "embed",
                      "all-minilm", "minilm", "e5", "gte")


def _auto(**kwargs):
    """Prefer a local Ollama embedding model, then sentence-transformers, then OpenAI, then hashing."""
    client = ollama.Client(kwargs.get("host"), timeout=3)
    if client.is_running():
        models = client.list_models()
        for pref in _OLLAMA_PREFERENCE:   # strongest general-purpose models first
            for m in models:
                if pref in m:
                    return OllamaEmbedder(m.removesuffix(":latest"), host=kwargs.get("host"))
    try:
        import sentence_transformers  # noqa: F401
        return SentenceTransformerEmbedder()
    except ImportError:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder()
        except ImportError:
            pass
    print("[chunking] no embedding model found; using the offline hashing embedder. "
          "Try: ollama pull nomic-embed-text", file=sys.stderr)
    return HashingEmbedder()


# Backwards-compatible helpers from v1.
def sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    return SentenceTransformerEmbedder(model_name)


def openai_embedder(model: str = "text-embedding-3-small") -> Embedder:
    return OpenAIEmbedder(model)


def embed_query(embed, text: str) -> List[float]:
    """Embed a search query, using the model's query prefix when it has one."""
    q = getattr(embed, "query", None)
    return q(text) if q else embed([text])[0]


# -- maths ----------------------------------------------------------------------------

def _normalize(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def mean(vectors: List[List[float]]) -> List[float]:
    n = len(vectors)
    return [sum(col) / n for col in zip(*vectors)]
