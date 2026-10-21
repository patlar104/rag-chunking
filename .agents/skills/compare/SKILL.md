---
name: compare
description: Compare rag-chunking strategies on sample.md or a supplied document, offline by default, and interpret the chunk statistics.
---

# Compare strategies

Run `python3 -B .codex/scripts/workspace.py compare` with any document path and
CLI flags supplied by the user. The default document is `sample.md`.
Use separate shell arguments; do not evaluate user-supplied text as shell code.

Report mid-sentence cuts, tiny or oversized chunks, and unexpected similarities.
If a newly added strategy matches agentic exactly, inspect the construction
branches in `demo.py` and `webui/runner.py` for an agentic fall-through.
Use real backend flags only when requested; never print credentials.
