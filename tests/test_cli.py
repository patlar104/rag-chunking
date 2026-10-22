"""End-to-end checks of the demo CLI (offline backends, no network)."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ["fixed", "recursive", "document", "semantic", "propositional", "agentic"]


def demo(*args, timeout=120):
    env = dict(
        os.environ, CHUNKING_CACHE_DIR=tempfile.mkdtemp(), OLLAMA_HOST="http://127.0.0.1:9", PYTHONIOENCODING="utf-8"
    )
    env.pop("OPENAI_API_KEY", None)
    env.pop("DEEPINFRA_API_KEY", None)
    return subprocess.run(
        [sys.executable, str(ROOT / "demo.py"), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        env=env,
        cwd=ROOT,
    )


class DemoCLI(unittest.TestCase):
    def test_compare_table_lists_every_strategy(self):
        r = demo("--compare")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("COMPARISON", r.stdout)
        for name in STRATEGIES:
            self.assertIn(name, r.stdout)

    def test_single_strategy_on_custom_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(
                "# Notes\n\nCats sleep a lot. They groom often.\n\n## Rust\n\nRust is memory safe. It compiles natively.\n"
            )
        try:
            r = demo(f.name, "-s", "document")
        finally:
            os.unlink(f.name)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("'headings': ['Notes', 'Rust']", r.stdout)
        self.assertIn("2 chunks", r.stdout)

    def test_query_retrieval(self):
        r = demo("-q", "how big is the offsite budget?")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("spending cap", r.stdout)

    def test_bad_backend_is_a_clean_error(self):
        r = demo("--embed", "bogus:thing", "--compare")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
