#!/usr/bin/env bash
# SessionStart hook: prepare a Claude Code cloud session so tests and lint work immediately.
# Remote-only (CLAUDE_CODE_REMOTE=true), synchronous, idempotent, non-interactive.
set -euo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export PIP_DISABLE_PIP_VERSION_CHECK=1

# Rebuild a missing venv or one whose interpreter is gone or older than 3.13,
# using the first interpreter on PATH that is 3.13 or newer.
if ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 13))' 2>/dev/null; then
  python_bin=
  for candidate in python3 python3.15 python3.14 python3.13; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 13))' 2>/dev/null; then
      python_bin=$candidate
      break
    fi
  done
  if [ -z "$python_bin" ]; then
    echo "error: Python 3.13 or newer is required to build .venv (python3 is $(python3 --version 2>&1))" >&2
    exit 1
  fi
  "$python_bin" -m venv --clear .venv
fi
.venv/bin/pip install --quiet -e ".[dev]"

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    printf 'export VIRTUAL_ENV=%q\n' "$PWD/.venv"
    printf 'export PATH=%q:"$PATH"\n' "$PWD/.venv/bin"
    echo 'export CHUNKING_CACHE_DIR=/tmp/chunk-cache'
  } >> "$CLAUDE_ENV_FILE"
fi
