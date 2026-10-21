---
name: add-backend
description: Add a new embedding or LLM backend (a new provider spec string such as `provider:model`) to rag-chunking. Use when wiring in another hosted API or local server for embeddings or chat, including getting prefixes, caching, events, JSON mode, tests and the dashboard picker right.
---

# Add an embedding or LLM backend

Backends are duck-typed callables, not subclasses of a shared interface:

- Embedder: `list[str] -> list[list[float]]`, optionally `.query(text)` for asymmetric models and a `.name`.
- LLM: `str -> str`, optionally `.complete_json(prompt, schema)` and a `.name`.

`chunking/deepinfra.py` is the closest template for a hosted OpenAI-style API; `chunking/ollama.py` for a local server.

## Checklist

1. **Client** in `chunking/<provider>.py` using only the standard library (`urllib`). Add retries for 429/5xx, a clear error type for "missing key", "not reachable" and "model not found", and read the API key from an environment variable. Never log or echo the key.
2. **Embedder class** extending `BaseEmbedder` in `chunking/embeddings.py`. It gets batching, normalisation and the automatic query/document prefixes (`KNOWN_PREFIXES`, keyed on model name). Add a prefix entry if the model needs one. Wrap each network call in `events.timed("embed.call", backend=..., n=...)`.
3. **LLM class** in `chunking/llm.py` (or the provider module, as DeepInfra does). Wrap calls in `events.timed("llm.call", backend=..., json=...)` and include token counts when the API returns them. Implement `complete_json` using the provider's JSON or structured-output mode, because the agentic and propositional chunkers depend on it through `ask_json`.
4. **Specs**: add the branch to `get_embedder()` (and `_auto()` if it should be auto-detected) and to `get_llm()`, and update both error messages that list valid specs. Keep `CachedEmbedder` wrapping consistent (`cache=True`, not for `hashing`).
5. **Dashboard**: `webui/server.py:_available()` decides which specs the picker offers; `webui/monitor.py` probes the embedder and LLM. Check that a new backend reports a dimension and latency. Public mode fixes backends at startup, so confirm the spend guard in `webui/runner.py` still counts its calls.
6. **Tests**: add a fake in-process `HTTPServer` (see `tests/test_backends.py`, `tests/test_deepinfra.py`) on `127.0.0.1` with port 0. Cover success, retry on 5xx, auth failure and bad JSON. Gate any real-network test behind an env var such as `CHUNKING_LIVE_<PROVIDER>=1`; tests must pass with no keys and no network.
7. **Docs**: README spec tables, the model-backend section and the Layout block; the optional extra in `pyproject.toml` and `requirements-optional.txt` only if a third-party package is needed.

## Verify

Run `$check`, then `python demo.py --doctor` to confirm detection. Do not run live-backend tests or put a real key in any file unless the user supplies one for this purpose.
