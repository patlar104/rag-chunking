---
name: check
description: Run rag-chunking's local CI checks and report lint, JavaScript syntax, test failures, and skipped optional integrations.
---

# Check the workspace

Run `python3 -B .codex/scripts/workspace.py check` from the repository root.
The runner prefers `.venv`, disables live backend tests, isolates the embedding
cache, and removes that cache after completion. It never installs packages.

Report each check, test count, skips, and relevant failure output. Identify missing
optional dependencies when tests skip; a skip is not a passing integration test.
If tooling is missing, report incomplete validation. Fix only failures within the
user's authorized task. For Python changes, lint the edited files before finishing;
for dashboard JavaScript changes, run the syntax check. These are explicit checks,
not an installed post-edit hook.
