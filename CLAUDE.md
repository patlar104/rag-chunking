# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rag-chunking`: six RAG chunking strategies (fixed, recursive, document, semantic, propositional, agentic), pluggable embedding/LLM backends, LangChain/LlamaIndex adapters, and a FastAPI "Chunk Monitor" dashboard. The core `chunking/` package and the Ollama/DeepInfra clients are **standard-library only** (`dependencies = []`); keep it that way. Anything needing a third-party package must be an optional extra and imported lazily.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                       # fastapi, httpx, langchain, llama-index, ruff, build

python -m unittest discover -s tests          # full suite (tests needing extras skip themselves)
python -m unittest tests.test_chunkers        # one module
python -m unittest tests.test_chunkers.Recursive.test_name   # one test (classes are plain TestCases, e.g. FixedSize, Recursive, Document)
ruff check .                                  # lint (CI also runs: node --check webui/static/app.js)

python demo.py --compare                      # offline comparison table on sample.md
python demo.py --doctor                       # which real backends are detected
python -m webui.server [--public] [--embed SPEC --llm SPEC]   # dashboard on :8000
CHUNKING_LIVE_OLLAMA=1 python -m unittest tests.test_backends        # opt-in live-backend tests
CHUNKING_LIVE_DEEPINFRA=1 python -m unittest tests.test_deepinfra
```

The owner's target is **Python 3.13 or newer (3.15 recommended)**; do not add compatibility shims for older versions, and use modern type syntax (`list[str]`, `X | None`, `collections.abc`). `import chunking` raises a clear `ImportError` below 3.13. `pyproject.toml` (`requires-python = ">=3.13"`), ruff (`py313`), the CI matrix (3.13–3.15) and the README all agree on this. Core tests run with zero dependencies (line length 120). The integration tests run with `-W error::ResourceWarning`, so close sockets/files. Set `CHUNKING_CACHE_DIR` to a temp dir when testing so the SQLite embedding cache doesn't touch `~/.cache/rag-chunking/`. Releases are cut by moving the `CHANGELOG.md` **Unreleased** items into a new `## [X.Y.Z]` section, bumping `version` in `pyproject.toml` to match, then pushing a `vX.Y.Z` tag; the Release workflow publishes that changelog section as the release notes and fails if the section is missing. `.python-version` (3.15) and `.mise.toml` both pin the local interpreter.

## Architecture

**Chunkers** (`chunking/*.py`) each expose `.split(text) -> list[Chunk]` and are registered in `STRATEGIES` in `chunking/__init__.py` (`get_chunker(name, **kwargs)`). A new strategy needs that registration plus a test in `tests/test_chunkers.py`; `demo.py:build()` and `webui/runner.py:_build()` also construct chunkers per strategy name, so check both. `Chunk` (`base.py`) carries `start`/`end` character offsets into the source plus a `metadata` dict. Fixed, recursive, semantic and agentic chunks satisfy `text[start:end] == chunk.text`; document chunks (heading-path prefix) and propositional chunks (rewritten facts, offsets point at the source passage) do not, so anything that re-slices the source (`with_overlap`, the integrations, the runner's boundary logic) must check or skip chunks that violate that. Token counts come from a simple regex word/punctuation tokenizer in `base.py` (swappable via `length_fn`).

**Backends are duck-typed callables, not base classes.** An embedder is `list[str] -> list[list[float]]` (optionally `.query()` for asymmetric models); an LLM is `str -> str` (optionally `.complete_json(prompt, schema)`). `get_embedder(spec)` / `get_llm(spec)` parse spec strings like `ollama:nomic-embed-text`, `deepinfra`, `st:...`, `openai:...`, `auto`. Embedder prefixes (`search_document:` / `search_query:`) are applied automatically per model name in `embeddings.py`. Real embedders can be wrapped in `CachedEmbedder` (SQLite). `ollama.py` and `deepinfra.py` are hand-rolled stdlib HTTP clients; `llm.py:ask_json` is the shared "ask for JSON, retry once, then raise" path.

**Every LLM-using chunker has an offline fallback.** `AgenticChunker` and `PropositionalChunker` fall back to rule-based heuristics when no LLM is given, or when the model's JSON fails after retries (emitting a `fallback` event). Semantic chunking defaults to a toy `HashingEmbedder`. This is why the whole suite and the live demo run with no network; keep new model-dependent code degrading the same way.

**Telemetry event bus** (`chunking/events.py`) is the seam between library and dashboard. Library code calls `events.emit(...)` / `events.timed(...)` (`embed.call`, `llm.call`, `agentic.decision`, `semantic.distances`, ...); it is nearly free when nothing subscribes. `current_run` is a contextvar that tags events with a run id. The dashboard is just a subscriber, so adding a new live visualisation means emitting an event in the chunker and handling it in `webui/server.py` (`FORWARD` / `on_event`) and `webui/static/app.js`.

**Web UI** (`webui/`): `server.py` holds module-level singletons (`hub` SSE fan-out, `metrics`, `runner`, `monitor`, `settings`, `PUBLIC`) and the REST + `/api/stream` SSE endpoints. `runner.py` is a single background worker thread that executes queued runs, wraps embedders/LLMs in `_Guard` (cancellation at the next model call plus the hourly LLM-call budget), and runs the optional ±15% boundary-stability check (`in_trial` suppresses live events during those re-runs). `monitor.py` does health probes and rolling stability metrics. The frontend is plain HTML/JS/SVG with no build step. **`--public` / `CHUNK_MONITOR_PUBLIC=1`** is the internet-facing mode (size/queue caps, backends fixed at startup, cancel tokens, probe cooldown, CORS allowlist, LLM spend guard); changes to the API surface need to be checked against it, and `tests/test_public_mode.py` covers it. The live demo is deployed from this mode with the DeepInfra backend.

**Integrations** (`chunking/integrations/`) adapt chunkers to LangChain (`ChunkingTextSplitter`, `ChunkingEmbeddings`) and LlamaIndex (`ChunkingNodeParser`, `ChunkingEmbedding`) in both directions (framework models can drive the chunkers, and our embedders can drive the frameworks). `_common.py` flattens chunk metadata to vector-store-safe scalars and lists `NON_EMBED_KEYS`, the metadata that must not be embedded.

## Conventions

- Tests are `unittest`, not pytest. Backend tests use fake in-process Ollama/DeepInfra HTTP servers instead of mocks; follow that pattern for new client code.
- Never read or print real API keys (`DEEPINFRA_API_KEY`, `OPENAI_API_KEY`); tests must not require them.
- Test counts and file listings in `README.md` ("75 tests") drift; update them if you add tests or modules.
- The `add-backend` and `add-chunking-strategy` skills are shared with Codex: the real files are in `.agents/skills/`, and `.claude/skills/` holds symlinks to them. Edit the `.agents` copy and keep the wording tool-neutral (`/check` in Claude Code, `$check` in Codex).
