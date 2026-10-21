#!/usr/bin/env bash
# PostToolUse hook for Edit/Write: lint the file that was just changed.
# Exits 2 (feeds stderr back to Claude) on a real lint failure; stays silent when the
# needed tool (ruff, node, jq) is not installed, so it never blocks work on a bare machine.
command -v jq >/dev/null 2>&1 || exit 0
file=$(jq -r '.tool_input.file_path // empty')
[ -n "$file" ] && [ -f "$file" ] || exit 0

case "$file" in
  *.py)
    command -v ruff >/dev/null 2>&1 || exit 0
    out=$(ruff check "$file" 2>&1) || { echo "$out" >&2; exit 2; }
    ;;
  */webui/static/app.js)
    command -v node >/dev/null 2>&1 || exit 0
    out=$(node --check "$file" 2>&1) || { echo "$out" >&2; exit 2; }
    ;;
esac
exit 0
