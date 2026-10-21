# Repository Guidelines

## Project Structure & Module Organization

- `chunking/` contains six chunking strategies, shared types, embedding/LLM backends, and telemetry. Framework adapters live in `chunking/integrations/`.
- `webui/` contains the FastAPI server, background runner, and health monitor. `webui/static/` holds plain HTML, JavaScript, and CSS; no frontend build is required.
- `demo.py` provides the CLI; `sample.md` is its default input. `tests/` contains automated tests; `docs/` holds documentation assets.
- `pyproject.toml` defines packaging, optional dependencies, and Ruff settings. `.github/workflows/` defines CI and releases.

## Build, Test, and Development Commands

Use Python 3.13+; `.python-version` and `.mise.toml` select 3.15 locally. Run commands from the repository root in an activated virtual environment.

```bash
pip install -e ".[dev]"                   # Install development extras
python demo.py --compare                 # Compare strategies offline
python -m webui.server                   # Serve dashboard at localhost:8000
python -m unittest discover -s tests -v   # Run tests; missing extras skip
ruff check .                            # Check Python lint and imports
node --check webui/static/app.js         # Check frontend JavaScript syntax
python -m build                         # Build wheel and sdist in dist/
```

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Follow existing annotations with modern syntax such as `list[str]` and `X | None`. Ruff targets Python 3.13 with a 120-character line length.

Keep the core dependency-free. Import optional packages lazily and preserve offline fallbacks. Register new chunkers in `chunking/__init__.py:STRATEGIES`; check construction paths in `demo.py` and `webui/runner.py`.

## Testing Guidelines

Use `unittest.TestCase`, `tests/test_*.py` files, and `test_*` methods. Add regression tests beside the affected feature; chunker tests belong in `tests/test_chunkers.py`. Backend tests use in-process fake HTTP servers. No numeric coverage threshold is configured.

Match integration CI with `python -W error::ResourceWarning -m unittest discover -s tests -v`. Close resources and direct `CHUNKING_CACHE_DIR` to an isolated temporary directory. Live backend tests require explicit opt-in through `CHUNKING_LIVE_OLLAMA` or `CHUNKING_LIVE_DEEPINFRA`.

## Commit & Pull Request Guidelines

History uses descriptive, imperative subjects such as “Add” and “Address”; follow that style. Keep changes focused. PRs should explain behavior, validation, and related issues; include screenshots for dashboard changes. Add an entry under **Unreleased** in `CHANGELOG.md`.

## Security & Configuration

Never commit or print API keys. Check API changes against `--public` protections and `tests/test_public_mode.py`. Public dashboard runs are visible to all viewers; avoid private input.

## Codex Workspace

See `.codex/README.md` for environment actions and `.agents/skills/` for development and review workflows. Prefer the existing `.venv`; run `python3 -B .codex/scripts/workspace.py check` for validation. Lint edited Python and syntax-check changed dashboard JavaScript before finishing. Preserve `.claude/` and `CLAUDE.md`. Leave changes uncommitted unless requested.
