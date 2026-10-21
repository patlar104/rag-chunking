---
description: Compare all six strategies on a document (offline by default)
argument-hint: [file] [extra demo.py flags]
allowed-tools: Bash(python demo.py:*), Bash(python3 demo.py:*)
---

Run `python demo.py --compare $ARGUMENTS` (the document defaults to `sample.md`; use `python3` if `python` is missing).

Then summarise the table: which strategy cut mid-sentence (`mid_sentence_cuts`), which produced tiny or oversized chunks, and anything that looks off compared with the README's sample (for example two strategies with identical numbers, which usually means one fell through to the agentic branch in `build()`).

Only add `--embed` or `--llm` flags if the user passed them. Never pass or print API keys. Real backends are optional, and the default run needs none.
