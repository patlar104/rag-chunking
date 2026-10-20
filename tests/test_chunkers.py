"""Run with:  python -m unittest discover -s tests   (or: pytest)"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking import (AgenticChunker, FixedSizeChunker, MarkdownChunker, PropositionalChunker,  # noqa: E402
                      RecursiveChunker, SemanticChunker, chunk_stats, count_tokens, get_chunker,
                      with_overlap)
from chunking.llm import parse_json  # noqa: E402

SAMPLE = (Path(__file__).resolve().parents[1] / "sample.md").read_text()
PROSE = ("Cats are small mammals. They sleep for most of the day. Cats groom themselves often. "
         "Rust is a systems programming language. It guarantees memory safety without a garbage collector. "
         "Rust programs compile to native code. The Rust borrow checker enforces ownership rules.")


def offsets_ok(tc, text, chunks):
    for c in chunks:
        tc.assertEqual(text[c.start:c.end], c.text, f"offsets don't match text for {c!r}")


class FixedSize(unittest.TestCase):
    def test_sizes_and_offsets(self):
        chunks = FixedSizeChunker(chunk_size=20).split(SAMPLE)
        self.assertTrue(all(c.tokens <= 20 for c in chunks))
        self.assertTrue(all(c.tokens == 20 for c in chunks[:-1]))
        offsets_ok(self, SAMPLE, chunks)

    def test_overlap_shares_tokens(self):
        chunks = FixedSizeChunker(chunk_size=20, overlap=5).split(PROSE)
        for a, b in zip(chunks, chunks[1:]):
            self.assertLess(b.start, a.end)  # windows overlap

    def test_chars_mode(self):
        chunks = FixedSizeChunker(chunk_size=50, overlap=10, unit="chars").split(PROSE)
        self.assertTrue(all(len(c.text) <= 50 for c in chunks))
        self.assertEqual(chunks[1].start, 40)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=10)


class Recursive(unittest.TestCase):
    def test_respects_limit_and_sentences(self):
        chunks = RecursiveChunker(chunk_size=25).split(PROSE)
        self.assertTrue(all(c.tokens <= 25 for c in chunks))
        self.assertTrue(all(c.text.endswith(".") for c in chunks))
        offsets_ok(self, PROSE, chunks)

    def test_falls_back_to_words(self):
        text = "word " * 100
        chunks = RecursiveChunker(chunk_size=10).split(text)
        self.assertTrue(all(c.tokens <= 10 for c in chunks))
        self.assertEqual(sum(c.tokens for c in chunks), 100)

    def test_small_text_single_chunk(self):
        self.assertEqual(len(RecursiveChunker(chunk_size=500).split(PROSE)), 1)


class Document(unittest.TestCase):
    def test_tables_and_code_kept_whole(self):
        chunks = MarkdownChunker(max_tokens=20).split(SAMPLE)
        tables = [c for c in chunks if "table" in c.metadata["blocks"]]
        code = [c for c in chunks if "code" in c.metadata["blocks"]]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].text.count("| Ready"), 3)       # all rows together
        self.assertIn("docker run", code[0].text)
        self.assertIn("redis-cli ping", code[0].text)              # whole code block

    def test_heading_path(self):
        chunks = MarkdownChunker().split(SAMPLE)
        table = next(c for c in chunks if "table" in c.metadata["blocks"])
        self.assertEqual(table.metadata["headings"], ["Redis Field Guide", "Installation", "Requirements"])
        self.assertTrue(table.text.startswith("Redis Field Guide > Installation"))

    def test_sections_dont_mix(self):
        for c in MarkdownChunker().split(SAMPLE):
            self.assertFalse("Eviction" in c.text and "offsite" in c.text)


class Semantic(unittest.TestCase):
    def test_topic_shift_is_cut(self):
        chunks = SemanticChunker(breakpoint_percentile=70).split(PROSE)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertFalse(any("Cats" in c.text and "Rust" in c.text for c in chunks))
        offsets_ok(self, PROSE, chunks)

    def test_custom_embedder(self):
        def embed(texts):  # 2-D toy embedding: is this about cats or rust?
            return [[t.lower().count("cat") + t.count("They"), t.lower().count("rust") + t.count("It ")
                     + t.count("borrow")] for t in texts]
        chunks = SemanticChunker(embed=embed, buffer=0, min_sentences=1).split(PROSE)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].text.startswith("Rust is"))


class Propositional(unittest.TestCase):
    def test_rule_based_resolves_pronouns(self):
        props = [c.text for c in PropositionalChunker().split(
            "Redis is fast. It stores data in RAM, and it supports many types.")]
        self.assertEqual(props, ["Redis is fast.", "Redis stores data in RAM.", "Redis supports many types."])

    def test_with_llm(self):
        fake = lambda prompt: '```json\n["Paris is the capital of France.", "Paris has 2 million people."]\n```'
        chunks = PropositionalChunker(llm=fake).split("Paris is France's capital, home to 2M people.")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["mode"], "llm")


class Agentic(unittest.TestCase):
    def test_heuristic_splits_topics(self):
        chunks = AgenticChunker().split(PROSE)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].text.startswith("Rust is"))

    def test_heuristic_breaks_on_headings(self):
        titles = [c.metadata["title"] for c in AgenticChunker().split(SAMPLE)]
        for t in ("Installation", "Persistence", "Eviction Policies", "Team Notes"):
            self.assertIn(t, titles)

    def test_with_llm(self):
        def fake(prompt):
            new = "Rust is" in prompt.split("Next sentence:")[1]
            return json.dumps({"boundary": new, "title": "Rust" if "Rust" in prompt else "Cats"})
        chunks = AgenticChunker(llm=fake).split(PROSE)
        self.assertEqual([c.metadata["title"] for c in chunks], ["Cats", "Rust"])

    def test_llm_garbage_does_not_crash(self):
        chunks = AgenticChunker(llm=lambda p: "no idea").split(PROSE)
        self.assertEqual(len(chunks), 1)

    def test_max_tokens(self):
        self.assertTrue(all(c.tokens <= 30 for c in AgenticChunker(max_tokens=30).split(SAMPLE)
                            if "```" not in c.text and "|" not in c.text))


class Helpers(unittest.TestCase):
    def test_with_overlap(self):
        base = RecursiveChunker(chunk_size=25).split(PROSE)
        over = with_overlap(PROSE, base, overlap_tokens=5)
        self.assertEqual(over[0].text, base[0].text)
        for b, o in zip(base[1:], over[1:]):
            self.assertTrue(o.text.endswith(b.text))
            self.assertEqual(count_tokens(o.text) - count_tokens(b.text), 5)
        offsets_ok(self, PROSE, over)

    def test_stats_flags_mid_sentence_cuts(self):
        fixed = chunk_stats(PROSE, FixedSizeChunker(chunk_size=12).split(PROSE))
        rec = chunk_stats(PROSE, RecursiveChunker(chunk_size=25).split(PROSE))
        self.assertGreater(fixed["mid_sentence_cuts"], 0)
        self.assertEqual(rec["mid_sentence_cuts"], 0)

    def test_registry(self):
        self.assertIsInstance(get_chunker("semantic"), SemanticChunker)
        with self.assertRaises(ValueError):
            get_chunker("nope")

    def test_parse_json(self):
        self.assertEqual(parse_json('Sure! {"boundary": true} hope that helps'), {"boundary": True})


if __name__ == "__main__":
    unittest.main()
