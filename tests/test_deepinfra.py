"""DeepInfra backend tests against a fake OpenAI-compatible server (no network, no key).
Set CHUNKING_LIVE_DEEPINFRA=1 (with DEEPINFRA_API_KEY) to also hit the real API."""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking import get_chunker, get_embedder, get_llm  # noqa: E402
from chunking.deepinfra import DeepInfraError  # noqa: E402

PROSE = (
    "Cats are small mammals. They sleep for most of the day. Cats groom themselves often. "
    "Rust is a systems programming language. It guarantees memory safety. Rust programs compile to native code."
)


class Fake(BaseHTTPRequestHandler):
    seen = []
    fail_next = 0

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if self.headers.get("Authorization") != "Bearer good" and self.headers.get("x-api-key") != "good":
            return self._send(401, {"error": {"message": "invalid api key"}})
        if Fake.fail_next:
            Fake.fail_next -= 1
            return self._send(503, {"detail": "busy"})
        if self.path.endswith("/embeddings"):
            vecs = []
            for i, t in enumerate(body["input"]):
                t = t.lower()
                vecs.append(
                    {
                        "index": i,
                        "embedding": [
                            t.count("cat") + t.count("they") + 0.01,
                            t.count("rust") + t.count("memory") + 0.01,
                            0.1,
                        ],
                    }
                )
            return self._send(200, {"data": list(reversed(vecs)), "usage": {"prompt_tokens": 5}})
        prompt = body["messages"][0]["content"]
        if body.get("response_format", {}).get("type") == "json_object":
            nxt = prompt.split("Next sentence:")[-1]
            reply = json.dumps({"boundary": "Rust is" in nxt, "title": "Rust" if "Rust" in nxt else "Cats"})
        else:
            reply = "OK"
        return self._send(
            200, {"choices": [{"message": {"content": reply}}], "usage": {"prompt_tokens": 50, "completion_tokens": 8}}
        )


class DeepInfraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}/v1/openai"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Fake.seen.clear()
        Fake.fail_next = 0

    def test_embeddings_order_prefix_and_semantic(self):
        emb = get_embedder("deepinfra", base_url=self.base, api_key="good")
        self.assertEqual(emb.name, "deepinfra:BAAI/bge-base-en-v1.5")
        vecs = emb(["cat", "rust"])
        self.assertGreater(vecs[0][0], vecs[0][1])  # reordered by index, normalised
        emb.query("memory")
        self.assertTrue(Fake.seen[-1]["body"]["input"][0].startswith("Represent this sentence"))
        chunks = get_chunker("semantic", embed=emb, buffer=1, min_sentences=1).split(PROSE)
        self.assertEqual(len(chunks), 2)

    def test_llm_json_mode_and_agentic(self):
        llm = get_llm("deepinfra", base_url=self.base, api_key="good")
        self.assertEqual(llm("hi"), "OK")
        chunks = get_chunker("agentic", llm=llm).split(PROSE)
        self.assertEqual([c.metadata["title"] for c in chunks], ["Cats", "Rust"])
        self.assertEqual(Fake.seen[-1]["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(Fake.seen[-1]["body"]["model"], "Qwen/Qwen2.5-7B-Instruct")

    def test_proxy_header_mode(self):
        llm = get_llm("deepinfra", base_url=self.base, api_key="good", auth_header="x-api-key")
        self.assertEqual(llm("hi"), "OK")
        self.assertNotIn("Authorization", Fake.seen[-1]["headers"])

    def test_retry_then_success_and_auth_error(self):
        Fake.fail_next = 1
        self.assertEqual(get_llm("deepinfra", base_url=self.base, api_key="good")("hi"), "OK")
        with self.assertRaises(DeepInfraError) as cm:
            get_llm("deepinfra", base_url=self.base, api_key="bad")("hi")
        self.assertEqual(cm.exception.status, 401)
        self.assertIn("invalid api key", str(cm.exception))
        self.assertNotIn("bad", str(cm.exception).replace("check", ""))  # never echo the key


class BudgetTests(unittest.TestCase):
    def test_llm_budget_blocks_after_limit(self):
        from webui.runner import Runner  # stdlib only, no FastAPI needed

        r = Runner(lambda *a: None)
        r.llm_budget = 2
        self.assertTrue(r._take_llm_call() and r._take_llm_call())
        self.assertFalse(r._take_llm_call())
        self.assertEqual(r.budget_state(), {"used": 2, "limit": 2})


@unittest.skipUnless(os.environ.get("CHUNKING_LIVE_DEEPINFRA") == "1", "set CHUNKING_LIVE_DEEPINFRA=1 for the live API")
class LiveDeepInfra(unittest.TestCase):
    def test_live(self):
        emb = get_embedder("deepinfra")
        self.assertEqual(len(emb(["hello"])[0]), 768)
        self.assertTrue(get_llm("deepinfra")("Reply with the word OK."))


if __name__ == "__main__":
    unittest.main()
