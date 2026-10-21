---
description: Run the same checks CI runs (ruff, JS syntax, unit tests)
allowed-tools: Bash(ruff check:*), Bash(node --check:*), Bash(python -m unittest:*), Bash(python3 -m unittest:*), Bash(git status:*)
---

Run the local equivalent of CI and report results, not just pass/fail:

1. `ruff check .`. If `ruff` is not installed, say so and skip; do not install it without asking.
2. `node --check webui/static/app.js`
3. `CHUNKING_CACHE_DIR=$(mktemp -d) python -m unittest discover -s tests`. Use `python3` if `python` is missing, and mention which interpreter version ran (the project targets 3.13+, so flag it if the interpreter is older).

Report the test count and the number skipped, and name which optional extras are missing when tests skip (FastAPI/httpx, LangChain, LlamaIndex). On a failure, show the failing test and the relevant traceback lines, then stop and ask before fixing anything unrelated to the current task.

$ARGUMENTS
