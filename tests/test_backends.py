"""Backend tests against a fake Ollama server (no real models or network needed).

Set CHUNKING_LIVE_OLLAMA=1 to also run the live tests against a real local Ollama.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking import AgenticChunker, PropositionalChunker, SemanticChunker, ollama  # noqa: E402
from chunking.embeddings import CachedEmbedder, HashingEmbedder, OllamaEmbedder, embed_query, get_embedder  # noqa: E402
from chunking.llm import OllamaLLM, ask_json, get_llm  # noqa: E402

PROSE = ("Cats are small mammals. They sleep for most of the day. Cats groom themselves often. "
         "Rust is a systems programming language. It guarantees memory safety without a garbage collector. "
         "Rust programs compile to native code. The Rust borrow checker enforces ownership rules.")


def toy_vector(text):
    t = text.lower()
    return [t.count("cat") + t.count("they") + t.count("mammal") + 0.01,
            t.count("rust") + t.count("memory") + t.count("borrow") + t.count("it ") + 0.01, 0.1]


class FakeOllama(BaseHTTPRequestHandler):
    installed = {"nomic-embed-text:latest", "qwen2.5:1.5b"}
    legacy = False          # simulate an old server without /api/embed
    requests = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj=None, raw=None):
        body = raw.encode() if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send(200, {"version": "0.0-test"})
        if self.path == "/api/tags":
            return self._send(200, {"models": [{"name": n} for n in sorted(self.installed)]})
        self._send(404, raw="404 page not found")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append((self.path, body))
        model = body.get("model", "")
        if self.path in ("/api/embed", "/api/embeddings", "/api/chat") and \
                model not in self.installed and f"{model}:latest" not in self.installed:
            return self._send(404, {"error": f'model "{model}" not found, try pulling it first'})
        if self.path == "/api/embed":
            if self.legacy:
                return self._send(404, raw="404 page not found")
            return self._send(200, {"embeddings": [toy_vector(t) for t in body["input"]]})
        if self.path == "/api/embeddings":
            return self._send(200, {"embedding": toy_vector(body["prompt"])})
        if self.path == "/api/chat":
            prompt = body["messages"][-1]["content"]
            if "propositions" in prompt:
                reply = {"propositions": ["Fact one.", "Fact two."]}
            else:
                nxt = prompt.split("Next sentence:")[1]
                reply = {"boundary": "Rust is" in nxt, "title": "Rust" if "Rust" in prompt else "Cats"}
            return self._send(200, {"message": {"role": "assistant", "content": json.dumps(reply)}})
        self._send(404, raw="404 page not found")


class WithFakeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.host = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeOllama.requests.clear()
        FakeOllama.legacy = False


class OllamaEmbedderTests(WithFakeServer):
    def test_batch_embed_and_prefixes(self):
        emb = OllamaEmbedder("nomic-embed-text", host=self.host)
        vecs = emb(["a cat", "rust code"])
        self.assertEqual(len(vecs), 2)
        self.assertAlmostEqual(sum(x * x for x in vecs[0]), 1.0, places=6)   # normalised
        path, body = FakeOllama.requests[-1]
        self.assertEqual(path, "/api/embed")
        self.assertEqual(body["input"][0], "search_document: a cat")          # nomic doc prefix
        embed_query(emb, "what is rust?")
        self.assertEqual(FakeOllama.requests[-1][1]["input"], ["search_query: what is rust?"])

    def test_legacy_endpoint_fallback(self):
        FakeOllama.legacy = True
        vecs = OllamaEmbedder("nomic-embed-text", host=self.host)(["x", "y", "z"])
        self.assertEqual(len(vecs), 3)
        self.assertEqual(FakeOllama.requests[-1][0], "/api/embeddings")

    def test_missing_model_message(self):
        with self.assertRaises(ollama.ModelNotFound) as ctx:
            OllamaEmbedder("mxbai-embed-large", host=self.host)(["x"])
        self.assertIn("ollama pull mxbai-embed-large", str(ctx.exception))

    def test_semantic_with_ollama(self):
        emb = OllamaEmbedder("nomic-embed-text", host=self.host, doc_prefix="")
        chunks = SemanticChunker(embed=emb, buffer=1, min_sentences=1).split(PROSE)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].text.startswith("Rust is"))

    def test_auto_prefers_ollama(self):
        emb = get_embedder("auto", host=self.host)
        self.assertEqual(emb.name, "ollama:nomic-embed-text")


class OllamaLLMTests(WithFakeServer):
    def test_structured_output_schema_sent(self):
        llm = OllamaLLM("qwen2.5:1.5b", host=self.host)
        data = ask_json(llm, "give propositions", {"type": "object"})
        self.assertEqual(data["propositions"], ["Fact one.", "Fact two."])
        body = FakeOllama.requests[-1][1]
        self.assertEqual(body["format"], {"type": "object"})
        self.assertEqual(body["options"]["temperature"], 0.0)

    def test_agentic_with_ollama(self):
        llm = OllamaLLM("qwen2.5:1.5b", host=self.host)
        chunks = AgenticChunker(llm=llm).split(PROSE)
        self.assertEqual([c.metadata["title"] for c in chunks], ["Cats", "Rust"])
        self.assertEqual(llm.calls, 6)  # one decision per sentence after the first

    def test_propositional_with_ollama(self):
        chunks = PropositionalChunker(llm=OllamaLLM("qwen2.5:1.5b", host=self.host)).split(PROSE)
        self.assertEqual(chunks[0].text, "Fact one.")
        self.assertEqual(chunks[0].metadata["mode"], "llm")

    def test_auto_llm_skips_embedding_models(self):
        self.assertEqual(get_llm("auto", host=self.host).model, "qwen2.5:1.5b")

    def test_missing_chat_model(self):
        with self.assertRaises(ollama.ModelNotFound):
            OllamaLLM("llama3.2:3b", host=self.host)("hi")


class ClientTests(unittest.TestCase):
    def test_not_running_message(self):
        c = ollama.Client("http://127.0.0.1:9", timeout=1)
        self.assertFalse(c.is_running())
        with self.assertRaises(ollama.OllamaNotRunning) as ctx:
            c.list_models()
        self.assertIn("ollama serve", str(ctx.exception))

    def test_host_resolution(self):
        self.assertEqual(ollama.resolve_host("0.0.0.0"), "http://127.0.0.1:11434")
        self.assertEqual(ollama.resolve_host("myserver:8080"), "http://myserver:8080")
        self.assertEqual(ollama.resolve_host("https://gpu.box/"), "https://gpu.box:11434")


class CacheTests(unittest.TestCase):
    def test_cache_hits(self):
        calls = []

        class Counting(HashingEmbedder):
            def _embed(self, texts):
                calls.extend(texts)
                return super()._embed(texts)

        with tempfile.TemporaryDirectory() as d:
            emb = CachedEmbedder(Counting(), path=os.path.join(d, "c.sqlite"))
            first = emb(["a", "b"])
            second = emb(["b", "a", "c"])
            self.assertEqual(calls, ["a", "b", "c"])
            self.assertEqual(second[1], first[0])
            self.assertEqual((emb.hits, emb.misses), (2, 3))
            emb.close()

    def test_get_embedder_specs(self):
        self.assertIsInstance(get_embedder("hashing"), HashingEmbedder)
        with self.assertRaises(ValueError):
            get_embedder("nope")


@unittest.skipUnless(os.environ.get("CHUNKING_LIVE_OLLAMA"), "set CHUNKING_LIVE_OLLAMA=1 to run")
class LiveOllama(unittest.TestCase):
    def test_live_embed_and_chat(self):
        emb = get_embedder(os.environ.get("CHUNKING_EMBED", "ollama:nomic-embed-text"))
        self.assertGreater(len(emb(["hello"])[0]), 100)
        llm = get_llm(os.environ.get("CHUNKING_LLM", "ollama:qwen2.5:1.5b"))
        data = ask_json(llm, 'Reply with JSON {"ok": true}', {"type": "object", "properties": {"ok": {"type": "boolean"}}})
        self.assertTrue(data["ok"])


if __name__ == "__main__":
    unittest.main()
