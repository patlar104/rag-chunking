"""Codex actions using the existing project venv; never install dependencies."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable


def run(args: list[str], env: dict[str, str]) -> int:
    print("+ " + " ".join(args), flush=True)
    return subprocess.run(args, cwd=ROOT, env=env, check=False).returncode


def doctor() -> int:
    # Run metadata inspection in the selected interpreter, without importing packages.
    code = """
import importlib.metadata
import shutil
import sys
print('Interpreter:', sys.executable)
print('Python:', sys.version.split()[0])
missing = []
for name in ['ruff', 'fastapi', 'uvicorn', 'httpx', 'langchain-core', 'langchain-text-splitters',
             'llama-index-core', 'build']:
    try:
        print(name + ': ' + importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        print(name + ': missing')
        missing.append(name)
if missing:
    print('Development extras are incomplete. No packages were installed.')
    print('After installation is authorized: python -m pip install -e ".[dev]"')
node = shutil.which('node')
print('Node:', node or 'missing (required for JavaScript syntax checks)')
sys.exit(0 if sys.version_info >= (3, 13) and not missing and node else 1)
"""
    return subprocess.run([PYTHON, "-B", "-c", code], cwd=ROOT, check=False).returncode


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"doctor", "check", "compare", "dashboard"}:
        print("Usage: workspace.py {doctor|check|compare|dashboard} [compare arguments]", file=sys.stderr)
        return 2
    action = sys.argv[1]
    if action != "compare" and len(sys.argv) != 2:
        print("Only compare accepts extra arguments.", file=sys.stderr)
        return 2
    if action == "doctor":
        return doctor()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", RUFF_NO_CACHE="true")
    # Keep the default checks offline even if the parent shell opts into live tests.
    for key in list(env):
        if key.startswith("CHUNKING_LIVE_"):
            env.pop(key)
    with tempfile.TemporaryDirectory(prefix="rag-chunking-codex-") as cache:
        env["CHUNKING_CACHE_DIR"] = cache
        print(f"Temporary cache: {cache} (removed on exit)", flush=True)
        if action == "compare":
            result = run([PYTHON, "-B", "demo.py", "--compare", *sys.argv[2:]], env)
        elif action == "dashboard":
            result = run([PYTHON, "-B", "-m", "webui.server"], env)
        else:
            results = [run([PYTHON, "-B", "-m", "ruff", "check", ".", ".codex/scripts"], env)]
            node = shutil.which("node")
            if node:
                results.append(run([node, "--check", "webui/static/app.js"], env))
            else:
                print("JavaScript syntax check unavailable: node is missing.", file=sys.stderr)
                results.append(1)
            results.append(
                run(
                    [
                        PYTHON,
                        "-B",
                        "-W",
                        "error::ResourceWarning",
                        "-m",
                        "unittest",
                        "discover",
                        "-s",
                        "tests",
                        "-v",
                    ],
                    env,
                )
            )
            result = int(any(results))
    print(f"Removed temporary cache: {cache}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
