# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is `0.x`, minor releases may include breaking changes, which are called out under **Changed** or **Removed**.

To cut a release, move the **Unreleased** items into a new version section, bump `version` in `pyproject.toml` to match, then push a tag (`git tag v0.5.1 && git push origin v0.5.1`). The Release workflow publishes that section as the GitHub Release notes.

## [Unreleased]

### Added
- Codex contributor guidelines, repository skills, and local environment actions for validation, offline comparisons, and the dashboard, using the existing project virtual environment.
- Claude Code cloud sessions run a `SessionStart` hook (`.claude/hooks/session-start.sh`) that creates `.venv` and installs `.[dev]`, so the full test suite and lint run immediately.
- `.coderabbit.yaml` with review settings for CodeRabbit, including path instructions for the stdlib-only core, `--public` mode and the test conventions.
- Issue forms for bug reports and feature requests, a label set in `.github/labels.yml`, and a `Repo metadata` workflow that syncs those labels and creates the `v0.5.1` and `v0.6.0` milestones when merged to `main`.

## [0.5.0] - 2026-10-20

### Changed
- **Python 3.13 is now the minimum version**, and 3.15 is recommended. CI tests on 3.13, 3.14 and 3.15 (Linux), plus 3.15 on macOS and Windows.
- Importing `chunking` on Python 3.12 or older now raises a clear `ImportError` that names the required version. pip also refuses to install on older versions (`requires-python = ">=3.13"`).
- The code now uses modern type syntax throughout (`list[str]`, `X | None`, `collections.abc`). Ruff targets `py313` with the pyupgrade rules on.
- The LlamaIndex extra no longer needs a Python-version marker.
- The Release workflow now uses this changelog for release notes, and fails if the tagged version has no section here.

### Added
- `CHANGELOG.md` (this file).
- A Contributors section in the README.
- `.python-version` (3.15), so `uv`, `pyenv` and editors pick the recommended interpreter.
- CI checks that the version in `pyproject.toml` has a changelog entry, and that Python 3.12 gets the clear "needs Python 3.13" error.

### Removed
- Support for Python 3.9–3.12.

## [0.4.0] - 2026-10-20

### Added
- **DeepInfra hosted backend** (`--embed deepinfra`, `--llm deepinfra`) using only the standard library. It defaults to `BAAI/bge-base-en-v1.5` (768-d, the closest hosted match for nomic-embed-text) and `Qwen/Qwen2.5-7B-Instruct`. It supports JSON mode, retries on 429 and 5xx, and can send the token through an auth proxy (`DEEPINFRA_BASE_URL`, `DEEPINFRA_AUTH_HEADER`).
- **Public mode** for the dashboard (`--public` or `CHUNK_MONITOR_PUBLIC=1`):
  - documents capped at 30k characters, and at most 3 queued runs
  - backends fixed at startup
  - duplicate strategies removed from each run
  - cancel tokens, so only the submitter can cancel a run
  - a probe cooldown, and a cap on live viewers
  - CORS limited to Perplexity-hosted origins
  - an hourly budget for LLM calls (`CHUNK_MONITOR_LLM_BUDGET`)
- A hosted-model panel in Model state, showing provider, dimension, probe latency, p95 and tokens/s.
- Live demo at [rag-chunking.pplx.app](https://rag-chunking.pplx.app), running in offline mode.
- `pyproject.toml` with optional extras: `webui`, `langchain`, `llamaindex`, `st`, `openai`, `dev`, `all`.
- MIT license.
- GitHub Actions:
  - **CI:** ruff, a test matrix, integrations, a dashboard smoke test, and a wheel build and install check
  - **CodeQL:** security scanning
  - **Release:** builds and publishes a GitHub Release on version tags
  - **Dependabot:** weekly dependency updates
- New tests for public mode and the demo CLI, plus fake-server tests for DeepInfra. That brings the suite to 75 tests.
- A README setup guide, a troubleshooting table and CI badges.

### Fixed
- A chart redraw race that threw `clientWidth of null` when you switched tabs mid-run.
- Agentic decision labels no longer repeat "continues".
- The Ollama row is no longer shown as an error when no backend uses Ollama.
- "Model calls" now matches the decision list for viewers who join mid-run. The count is tracked on the server.
- CodeQL findings: a possible `raise None` in `ask_json`, plus clarified best-effort `except` blocks.

## [0.3.0] - 2026-10-20

### Added
- **LangChain integration:**
  - `ChunkingTextSplitter`, which uses any strategy as a LangChain `TextSplitter`
  - `ChunkingEmbeddings`, a LangChain `Embeddings` wrapper
  - adapters that use LangChain embedders and chat models inside the chunkers
- **LlamaIndex integration:**
  - `ChunkingNodeParser`, with character offsets, relationships and metadata kept out of the embedding
  - `ChunkingEmbedding`
  - adapters for LlamaIndex embedders and LLMs
- **Chunk Monitor** web dashboard (FastAPI + Server-Sent Events):
  - **Health:** Ollama, embedder, LLM, cache, worker and process
  - **Stability:** healthy share, status changes, error and retry rates, latency jitter, and p50/p95 charts
  - **Model state:** what's loaded and when it unloads
  - **Live view:** for every strategy, including the agentic decision loop and the semantic distance curve
- **Boundary stability check:** re-runs each strategy with the size changed by ±15% and scores how many cut points stay put.
- Telemetry event bus (`chunking.events`) that the backends and chunkers report to.

### Fixed
- Markdown chunk offsets now point exactly into the source text.

## [0.2.0] - 2026-10-20

### Added
- Real embedding models: Ollama (`ollama:<model>`), sentence-transformers (`st:<model>`), OpenAI, and `auto` (the best one available).
- Real LLMs for the propositional and agentic chunkers: Ollama, OpenAI and `auto`. They use structured JSON output, retry once on bad replies, then fall back to the offline rules.
- An Ollama client that uses only the standard library: batched `/api/embed`, host selection with `OLLAMA_HOST`, and clear errors.
- Automatic query/document prefixes for asymmetric embedding models (nomic, e5, bge, mxbai).
- A SQLite embedding cache. A repeat run on the sample went from about 4 s to 0.1 s.
- `demo.py --doctor` (shows what's installed and what `auto` would pick) and `--pull` (downloads missing Ollama models).
- Backend tests against a fake Ollama server, for 38 tests in total.

### Changed
- The propositional chunker no longer sends tables or code blocks to the model.
- The agentic chunker opens a new chunk at headings without calling the model.

## [0.1.0] - 2026-10-20

### Added
- The six chunking strategies from "The 6 Production Chunking Algorithms for RAG": fixed-size, recursive, document-based (Markdown-aware), semantic, propositional and agentic (inspect → contextualize → boundary).
- `with_overlap()`, the video's "practical fix": 20–50 tokens of sliding-window overlap, for any chunker.
- `chunk_stats()` diagnostics for the video's Problem #1 (mid-sentence cuts) and Problem #2 (oversized chunks).
- `demo.py`, which compares every strategy on `sample.md` and finds the best chunk for a question (`-q`).
- Offline defaults: a hashing embedder and rule-based fallbacks, so it runs without installing anything.
- 23 unit tests.

[Unreleased]: https://github.com/patlar104/rag-chunking/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/patlar104/rag-chunking/releases/tag/v0.5.0
[0.4.0]: https://github.com/patlar104/rag-chunking/tree/f8814b0
[0.3.0]: https://github.com/patlar104/rag-chunking/tree/4b80efd
