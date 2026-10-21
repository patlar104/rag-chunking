"""Hosted models through DeepInfra (OpenAI-compatible API), standard library only.

    export DEEPINFRA_API_KEY=...            # https://deepinfra.com/dash/api_keys
    get_embedder("deepinfra")              # BAAI/bge-base-en-v1.5, 768-d, ~$0.005 / 1M tokens
    get_llm("deepinfra")                   # Qwen/Qwen2.5-7B-Instruct, ~$0.04 in / $0.10 out per 1M tokens

Why these defaults: DeepInfra doesn't serve nomic-embed-text or qwen2.5:1.5b, so we use
the closest hosted equivalents. bge-base has the same 768 dimensions as nomic; Qwen2.5-7B is
the same family as the local 1.5B model, and much better at agentic boundary decisions.

Configuration (environment):
    DEEPINFRA_API_KEY      token (sent as `Authorization: Bearer ...` by default)
    DEEPINFRA_BASE_URL     default https://api.deepinfra.com/v1/openai (point at a proxy if needed)
    DEEPINFRA_AUTH_HEADER  send the token in this header instead, e.g. "x-api-key" for an auth proxy
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from . import events
from .embeddings import BaseEmbedder

DEFAULT_BASE = "https://api.deepinfra.com/v1/openai"
DEFAULT_EMBED = "BAAI/bge-base-en-v1.5"
DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"


class DeepInfraError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def configured() -> bool:
    """True when a key (or an auth proxy) is configured."""
    return bool(os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("DEEPINFRA_BASE_URL"))


class Client:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 auth_header: Optional[str] = None, timeout: float = 60, retries: int = 2):
        self.base = (base_url or os.environ.get("DEEPINFRA_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.key = api_key if api_key is not None else os.environ.get("DEEPINFRA_API_KEY", "")
        self.auth_header = auth_header or os.environ.get("DEEPINFRA_AUTH_HEADER") or "Authorization"
        self.timeout, self.retries = timeout, retries

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "rag-chunking"}
        if self.key:
            h[self.auth_header] = f"Bearer {self.key}" if self.auth_header.lower() == "authorization" else self.key
        return h

    def post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base + path, data=data, headers=self._headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                detail = _error_detail(e)
                e.close()
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    events.emit("llm.retry", backend="deepinfra", reason=f"HTTP {e.code}: {detail}"[:200])
                    time.sleep(1.5 * (attempt + 1))
                    continue
                hint = " (check DEEPINFRA_API_KEY)" if e.code in (401, 403) else ""
                raise DeepInfraError(f"DeepInfra HTTP {e.code}: {detail}{hint}", e.code) from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise DeepInfraError(f"Can't reach DeepInfra at {self.base}: {e}") from None
        raise DeepInfraError("unreachable")  # pragma: no cover


def _error_detail(e: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(e.read().decode() or "{}")
        err = body.get("error", body.get("detail", body))
        if isinstance(err, dict):
            err = err.get("message", err)
        return str(err)[:300]
    except Exception:
        return e.reason or ""


class DeepInfraEmbedder(BaseEmbedder):
    """Hosted embedding model. Query prefixes for bge/e5 are applied automatically."""

    batch_size = 64

    def __init__(self, model: str = DEFAULT_EMBED, doc_prefix: Optional[str] = None,
                 query_prefix: Optional[str] = None, **client_kw: Any):
        super().__init__(doc_prefix, query_prefix, model)
        self.model, self.name = model, f"deepinfra:{model}"
        self.client = Client(**client_kw)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        r = self.client.post("/embeddings", {"model": self.model, "input": texts, "encoding_format": "float"})
        return [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]


class DeepInfraLLM:
    """Hosted chat model; `.complete_json` uses JSON mode."""

    def __init__(self, model: str = DEFAULT_LLM, temperature: float = 0.0, max_tokens: int = 512,
                 **client_kw: Any):
        self.model, self.name = model, f"deepinfra:{model}"
        self.temperature, self.max_tokens = temperature, max_tokens
        self.client = Client(**client_kw)
        self.calls = 0

    def _chat(self, prompt: str, json_mode: bool = False) -> str:
        body: Dict[str, Any] = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        self.calls += 1
        t0 = time.perf_counter()
        with events.timed("llm.call", backend=self.name, json=json_mode, prompt_chars=len(prompt)) as info:
            r = self.client.post("/chat/completions", body)
            usage = r.get("usage") or {}
            if usage:
                info.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
                secs = time.perf_counter() - t0
                if usage.get("completion_tokens") and secs > 0:
                    info["tokens_per_s"] = round(usage["completion_tokens"] / secs, 1)
        return (r["choices"][0]["message"].get("content") or "").strip()

    def __call__(self, prompt: str) -> str:
        return self._chat(prompt)

    def complete_json(self, prompt: str, schema: Optional[dict] = None) -> str:
        return self._chat(prompt, json_mode=True)

    def __repr__(self):
        return f"<DeepInfraLLM {self.model}>"
