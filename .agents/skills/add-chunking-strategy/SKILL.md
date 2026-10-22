---
name: add-chunking-strategy
description: Add a new chunking strategy to rag-chunking end to end. Use when the user wants a new chunker, a seventh strategy, or to rename or remove an existing one, since a strategy must be wired into several files that fail silently if one is missed.
---

# Add a chunking strategy

A strategy is not finished when `STRATEGIES` has an entry. Two `_build` functions end in an unconditional fall-through to `AgenticChunker`, so a strategy registered only in `STRATEGIES` runs as **agentic** in the CLI and the dashboard with no error.

## Checklist

1. **Chunker class** in `chunking/<name>.py`, exposing `.split(text) -> list[Chunk]`.
   - Standard library only. A third-party import belongs behind a lazy import and an optional extra in `pyproject.toml`.
   - Take `embed` or `llm` as plain callables (embedders accept a list of texts; LLMs accept a prompt string). Accept `**_ignored` if it should tolerate extra kwargs like the existing ones.
   - Offsets: `Chunk.start`/`end` index the source text. If `chunk.text` is not `text[start:end]` (prefixes, rewritten text), say so in the class docstring. The runner, `with_overlap` and the integrations check this.
   - If it uses a model, it needs an offline fallback and must `events.emit("fallback", strategy=..., reason=...)` when it degrades.
   - Emit `events.*` for anything the dashboard should show live, and document new event types in the `chunking/events.py` module docstring.
2. **Register** it in `STRATEGIES` and `__all__` handling in `chunking/__init__.py`.
3. **`demo.py:build()`**: add an explicit `if name == "<name>":` branch **above** the final agentic `return`.
4. **`webui/runner.py:_build()`**: same, add the branch above the final agentic `return`. If it calls an LLM, add the name to `LLM_STRATEGIES` (this decides whether an LLM backend is built and the spend budget applies). Decide whether `_chunks()` should apply `with_overlap` for it, and whether `_stability()` should skip it (propositional is skipped).
5. **Dashboard frontend**: add a label to `STRATEGY_INFO` in `webui/static/app.js`, and to `LLM_STRATS` if it needs an LLM. A live-trace view needs an event handler plus a branch in the tab rendering. Run `node --check webui/static/app.js`.
6. **Tests** in `tests/test_chunkers.py`: one `unittest.TestCase` class per strategy, covering sizes, offsets (or the documented exception), empty input and the offline fallback. `tests/test_webui.py` asserts the exact strategy list returned by `/api/config` (around line 102), so update that expectation and run the file.
7. **Docs**: README strategy table, "The strategies" section, the Layout block and the test count ("75 tests" appears in several places) if they changed.

## Verify

Use the repo check (`/check` in Claude Code, `$check` in Codex), then `python demo.py --compare` and confirm the new row differs from `agentic` (the fall-through bug shows up as identical numbers). With the web extras installed, also run `python -m unittest tests.test_webui tests.test_public_mode`.
