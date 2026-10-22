"""Web UI backend tests: metrics, stability scoring, worker, API (offline, no Ollama needed)."""

import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking import events  # noqa: E402

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
class MonitorPieces(unittest.TestCase):
    def test_boundary_agreement(self):
        from webui.runner import boundary_agreement

        self.assertEqual(boundary_agreement([100, 200], [100, 200]), 1.0)
        self.assertEqual(boundary_agreement([100, 200], [110, 190]), 1.0)  # within tolerance
        self.assertEqual(boundary_agreement([100, 200], [500]), 0.0)
        self.assertAlmostEqual(boundary_agreement([100, 200], [100]), 0.6667, places=3)
        self.assertEqual(boundary_agreement([], []), 1.0)

    def test_metrics_percentiles_errors_and_stability(self):
        from webui.monitor import Metrics

        m = Metrics()
        now = time.time()
        for ms in [10, 20, 30, 40, 1000]:
            m.record({"type": "llm.call", "ts": now, "run": "r1", "backend": "x", "ms": ms, "ok": True})
        m.record({"type": "llm.call", "ts": now, "run": "r1", "backend": "x", "ms": 5, "ok": False, "error": "boom"})
        m.record({"type": "llm.retry", "ts": now, "run": "r1"})
        m.record({"type": "fallback", "ts": now, "run": "r1", "strategy": "propositional"})
        m.record({"type": "cache.lookup", "ts": now, "hits": 3, "misses": 1})
        m.record({"type": "embed.call", "ts": now, "run": "probe", "backend": "x", "ms": 9, "ok": True})
        s = m.snapshot()
        llm = s["ops"]["llm"]
        self.assertEqual((llm["calls_5m"], llm["errors_5m"]), (6, 1))
        self.assertEqual(llm["p50"], 30)
        self.assertEqual(llm["last_error"]["error"], "boom")
        self.assertEqual(s["ops"]["embed"]["calls_5m"], 0)  # health probes don't pollute work metrics
        self.assertEqual(s["cache"]["hit_rate"], 0.75)
        self.assertEqual((s["stability"]["retries_5m"], s["stability"]["fallbacks_5m"]), (1, 1))
        self.assertEqual(len(s["series"]), 120)
        self.assertEqual(s["series"][-1]["calls"], 6)

    def test_runner_runs_cancels_and_reports(self):
        from webui.runner import Runner

        published = []
        r = Runner(lambda kind, data: published.append(kind))
        run = r.submit(SAMPLE, ["fixed", "document", "semantic", "agentic"], 60, 10, "hashing", "none", True)
        self.assertTrue(wait_for(lambda: r.runs[run["id"]]["status"] == "done"))
        d = r.detail(run["id"])
        self.assertEqual(set(d["results"]), {"fixed", "document", "semantic", "agentic"})
        for name, res in d["results"].items():
            self.assertEqual(res["status"], "done", name)
            self.assertGreater(res["stats"]["chunks"], 1)
            self.assertIn("score", res["stability"])
        self.assertIn("strategy.finished", published)
        self.assertEqual(r.state()["completed"], 1)
        with self.assertRaises(ValueError):
            r.submit("text", ["nope"], 60, 0, "hashing", "none", False)

        slow = r.submit(SAMPLE * 3, ["agentic"], 60, 0, "hashing", "none", False)
        r.cancel(slow["id"])
        self.assertTrue(wait_for(lambda: r.runs[slow["id"]]["status"] in ("cancelled", "done")))


@unittest.skipUnless(HAVE_WEB, "pip install fastapi uvicorn httpx")
class Api(unittest.TestCase):
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

    def test_config_and_static(self):
        c = self.client.get("/api/config").json()
        self.assertEqual(c["strategies"], ["fixed", "recursive", "document", "semantic", "propositional", "agentic"])
        self.assertIn("redis-guide", c["samples"])
        html = self.client.get("/").text
        self.assertIn("__PORT_8000__", html)
        self.assertEqual(self.client.get("/app.js").status_code, 200)

    def test_health_snapshot(self):
        self.assertTrue(wait_for(lambda: self.server.monitor.snapshot.get("overall") is not None))
        h = self.client.get("/api/health").json()
        self.assertEqual(set(h["components"]), {"ollama", "embedder", "llm", "cache", "worker", "process", "hosted"})
        self.assertEqual(h["components"]["embedder"]["status"], "ok")  # hashing embedder always works
        self.assertEqual(h["components"]["llm"]["status"], "off")

    def test_run_lifecycle_and_live_trace(self):
        r = self.client.post("/api/runs", json={"text": SAMPLE, "strategies": ["recursive", "agentic"], "size": 60})
        self.assertEqual(r.status_code, 201)
        rid = r.json()["id"]
        self.assertTrue(wait_for(lambda: self.client.get(f"/api/runs/{rid}").json()["status"] == "done"))
        d = self.client.get(f"/api/runs/{rid}").json()
        self.assertGreater(d["results"]["agentic"]["stats"]["chunks"], 2)
        self.assertGreater(len(d["live"]["agentic"]["decisions"]), 5)  # late viewers can replay
        self.assertTrue(any(x["id"] == rid for x in self.client.get("/api/runs").json()))
        self.assertGreater(self.client.get("/api/metrics").json()["totals"]["embed"], -1)

    def test_errors_are_4xx(self):
        self.assertEqual(self.client.post("/api/runs", json={"text": "x", "strategies": ["bogus"]}).status_code, 400)
        self.assertEqual(self.client.post("/api/runs", json={"text": "  ", "strategies": ["fixed"]}).status_code, 400)
        self.assertEqual(
            self.client.post("/api/runs", json={"text": "x", "strategies": ["fixed"], "size": 1}).status_code, 422
        )
        self.assertEqual(self.client.get("/api/runs/nope").status_code, 404)
        self.assertEqual(self.client.post("/api/runs/nope/cancel").status_code, 409)
        self.assertEqual(self.client.post("/api/settings", json={"embed": "bogus:thing"}).status_code, 400)


class EventBus(unittest.TestCase):
    def test_noop_without_subscribers_and_tags_run(self):
        events.emit("x")  # must not raise with no subscribers
        seen = []
        unsub = events.subscribe(seen.append)
        token = events.current_run.set("r9")
        try:
            with events.timed("op", a=1) as info:
                info["b"] = 2

            def failing_op():
                with events.timed("op"):
                    raise RuntimeError("bad")

            self.assertRaises(RuntimeError, failing_op)
        finally:
            events.current_run.reset(token)
            unsub()
        self.assertEqual((seen[0]["run"], seen[0]["ok"], seen[0]["a"], seen[0]["b"]), ("r9", True, 1, 2))
        self.assertFalse(seen[1]["ok"])
        self.assertIn("bad", seen[1]["error"])
        self.assertEqual(json.loads(json.dumps(seen[0]))["type"], "op")


if __name__ == "__main__":
    unittest.main()
