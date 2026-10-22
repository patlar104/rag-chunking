#!/usr/bin/env bash
# SessionStart hook: prepare a Claude Code cloud session so tests and lint work immediately.
# Remote-only (CLAUDE_CODE_REMOTE=true), synchronous, idempotent, non-interactive.
set -euo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 13))'; then
  echo "warning: $(python3 --version) is older than the required Python 3.13" >&2
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1

# Rebuild a missing venv or one whose interpreter is gone or older than 3.13.
if ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 13))' 2>/dev/null; then
  python3 -m venv --clear .venv
fi
.venv/bin/pip install --quiet -e ".[dev]"

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export VIRTUAL_ENV=\"$PWD/.venv\""
    echo "export PATH=\"$PWD/.venv/bin:\$PATH\""
    echo 'export CHUNKING_CACHE_DIR=/tmp/chunk-cache'
  } >> "$CLAUDE_ENV_FILE"
fi
