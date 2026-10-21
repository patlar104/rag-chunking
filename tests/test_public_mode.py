"""Public (internet-facing) mode: size/queue limits, fixed backends, cancel tokens, rate limits."""
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    HAVE_WEB = True
except ImportError:
    HAVE_WEB = False

SAMPLE = (Path(__file__).resolve().parents[1] / "sample.md").read_text()


def wait_for(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


@unittest.skipUnless(HAVE_WEB, "pip install fastapi uvicorn httpx")
class PublicMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from webui import server
        cls.server = server
        cls.cm = TestClient(server.app)
        cls.client = cls.cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.cm.__exit__(None, None, None)

    def setUp(self):
        self.saved = dict(self.server.PUBLIC)
        self.server.PUBLIC.update(enabled=True, last_probe=0)

    def tearDown(self):
        self.server.PUBLIC.clear()
        self.server.PUBLIC.update(self.saved)

    def run_and_wait(self, **body):
        r = self.client.post("/api/runs", json={"text": SAMPLE[:2000], "strategies": ["fixed"], **body})
        self.assertEqual(r.status_code, 201, r.text)
        rid = r.json()["id"]
        self.assertTrue(wait_for(lambda: self.client.get(f"/api/runs/{rid}").json()["status"] in ("done", "cancelled")))
        return r.json()

    def test_config_advertises_limits(self):
        limits = self.client.get("/api/config").json()["limits"]
        self.assertTrue(limits["public"])
        self.assertEqual(limits["max_chars"], 30_000)

    def test_document_size_cap(self):
        r = self.client.post("/api/runs", json={"text": "word " * 7000, "strategies": ["fixed"]})
        self.assertEqual(r.status_code, 413)

    def test_queue_cap(self):
        with mock.patch.object(self.server.runner, "state", return_value={"queued": 3}):
            r = self.client.post("/api/runs", json={"text": "Hello there.", "strategies": ["fixed"]})
        self.assertEqual(r.status_code, 429)

    def test_backends_are_fixed(self):
        r = self.client.post("/api/settings", json={"embed": "hashing", "llm": "none"})
        self.assertEqual(r.status_code, 403)

    def test_duplicate_strategies_collapse(self):
        out = self.run_and_wait(strategies=["fixed", "fixed", "recursive", "fixed"])
        self.assertEqual(out["strategies"], ["fixed", "recursive"])

    def test_cancel_token_required_and_never_leaked(self):
        out = self.run_and_wait()
        token, rid = out["cancel_token"], out["id"]
        self.assertTrue(token)
        self.assertNotIn(token, self.client.get(f"/api/runs/{rid}").text)
        self.assertNotIn(token, self.client.get("/api/runs").text)
        self.assertEqual(self.client.post(f"/api/runs/{rid}/cancel").status_code, 403)
        self.assertEqual(self.client.post(f"/api/runs/{rid}/cancel", json={"token": "wrong"}).status_code, 403)
        # right token passes the ownership check (the run already finished, so it's a 409, not a 403)
        self.assertEqual(self.client.post(f"/api/runs/{rid}/cancel", json={"token": token}).status_code, 409)

    def test_probe_cooldown(self):
        self.assertEqual(self.client.post("/api/probe").status_code, 200)
        self.assertEqual(self.client.post("/api/probe").status_code, 429)

    def test_private_mode_is_unrestricted(self):
        self.server.PUBLIC["enabled"] = False
        self.assertIsNone(self.client.get("/api/config").json()["limits"])
        self.assertEqual(self.client.post("/api/settings", json={"embed": "hashing", "llm": "none"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
