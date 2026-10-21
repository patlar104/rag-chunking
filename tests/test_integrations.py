"""LangChain / LlamaIndex integration tests (skipped if the framework isn't installed)."""
import json
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=UserWarning)

from chunking import events  # noqa: E402

SAMPLE = (Path(__file__).resolve().parents[1] / "sample.md").read_text()
PROSE = ("Cats are small mammals. They sleep for most of the day. Cats groom themselves often. "
         "Rust is a systems programming language. It guarantees memory safety without a garbage collector. "
         "Rust programs compile to native code. The Rust borrow checker enforces ownership rules.")


def toy(text):
    t = text.lower()
    return [t.count("cat") + t.count("they") + 0.01, t.count("rust") + t.count("memory") + 0.01, 0.1]


def boundary_reply(prompt):
    new = "Rust is" in prompt.split("Next sentence:")[1]
    return json.dumps({"boundary": new, "title": "Rust" if "Rust" in prompt else "Cats"})


try:
    import langchain_core  # noqa: F401
    import langchain_text_splitters  # noqa: F401
    HAVE_LC = True
except ImportError:
    HAVE_LC = False

try:
    import llama_index.core  # noqa: F401
    HAVE_LI = True
except ImportError:
    HAVE_LI = False


@unittest.skipUnless(HAVE_LC, "pip install langchain-core langchain-text-splitters")
class LangChainTests(unittest.TestCase):
    def setUp(self):
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        from chunking.integrations.langchain import ChunkingEmbeddings, ChunkingTextSplitter
        self.Document, self.Splitter, self.CE = Document, ChunkingTextSplitter, ChunkingEmbeddings

        class ToyEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [toy(t) for t in texts]

            def embed_query(self, text):
                return toy(text)

        class PromptAwareChat(GenericFakeChatModel):
            def invoke(self, input, *a, **k):  # reply depends on the prompt
                from langchain_core.messages import AIMessage
                return AIMessage(content=boundary_reply(input if isinstance(input, str) else str(input)))

        self.ToyEmbeddings, self.Chat = ToyEmbeddings, PromptAwareChat

    def test_document_splitter_metadata(self):
        docs = self.Splitter("document", max_tokens=80).split_documents(
            [self.Document(page_content=SAMPLE, metadata={"source": "sample.md"})])
        self.assertGreater(len(docs), 4)
        table = next(d for d in docs if "table" in d.metadata["blocks"])
        self.assertEqual(table.metadata["headings"], "Redis Field Guide > Installation > Requirements")
        self.assertEqual(table.metadata["source"], "sample.md")
        for d in docs:  # vector-store safe + real offsets
            self.assertTrue(all(isinstance(v, (str, int, float, bool)) for v in d.metadata.values()))
            self.assertEqual(SAMPLE[d.metadata["start_index"]:d.metadata["end_index"]].strip()[-20:],
                             d.page_content[-20:])

    def test_split_text_and_overlap(self):
        plain = self.Splitter("recursive", chunk_size=25).split_text(PROSE)
        over = self.Splitter("recursive", chunk_size=25, overlap_tokens=5).split_text(PROSE)
        self.assertEqual(len(plain), len(over))
        self.assertTrue(over[1].endswith(plain[1]) and len(over[1]) > len(plain[1]))

    def test_semantic_with_langchain_embeddings(self):
        chunks = self.Splitter("semantic", embeddings=self.ToyEmbeddings(), buffer=1, min_sentences=1).split_text(PROSE)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].startswith("Rust is"))

    def test_agentic_with_langchain_chat_model(self):
        llm = self.Chat(messages=iter([]))
        docs = self.Splitter("agentic", llm=llm).create_documents([PROSE])
        self.assertEqual([d.metadata["title"] for d in docs], ["Cats", "Rust"])

    def test_chunking_embeddings_in_vector_store(self):
        from langchain_core.vectorstores import InMemoryVectorStore
        store = InMemoryVectorStore(self.CE("hashing"))
        store.add_documents(self.Splitter("document", max_tokens=80).create_documents([SAMPLE]))
        hit = store.similarity_search("spending cap for each team at the offsite", k=1)[0]
        self.assertIn("Team Notes", hit.metadata["headings"])

    def test_events_from_wrapped_models(self):
        seen = []
        unsub = events.subscribe(seen.append)
        try:
            self.Splitter("semantic", embeddings=self.ToyEmbeddings()).split_text(PROSE)
        finally:
            unsub()
        calls = [e for e in seen if e["type"] == "embed.call"]
        self.assertTrue(calls and calls[0]["backend"].startswith("langchain:") and calls[0]["ok"])


@unittest.skipUnless(HAVE_LI, "pip install llama-index-core")
class LlamaIndexTests(unittest.TestCase):
    def setUp(self):
        from llama_index.core import Document
        from llama_index.core.base.embeddings.base import BaseEmbedding
        from llama_index.core.llms import CompletionResponse
        from llama_index.core.llms.mock import MockLLM

        from chunking.integrations.llama_index import ChunkingEmbedding, ChunkingNodeParser
        self.Document, self.Parser, self.CE = Document, ChunkingNodeParser, ChunkingEmbedding

        class ToyEmbedding(BaseEmbedding):
            def _get_text_embedding(self, text):
                return toy(text)

            def _get_query_embedding(self, query):
                return toy(query)

            async def _aget_query_embedding(self, query):
                return toy(query)

        class PromptAwareLLM(MockLLM):
            def complete(self, prompt, formatted=False, **kw):
                return CompletionResponse(text=boundary_reply(prompt))

        self.ToyEmbedding, self.LLM = ToyEmbedding, PromptAwareLLM

    def test_node_parser_offsets_metadata_relationships(self):
        from llama_index.core.schema import MetadataMode, NodeRelationship
        nodes = self.Parser("document", max_tokens=80).get_nodes_from_documents(
            [self.Document(text=SAMPLE, metadata={"source": "sample.md"})])
        self.assertGreater(len(nodes), 4)
        n = nodes[1]
        self.assertEqual(n.metadata["source"], "sample.md")
        self.assertIn(NodeRelationship.SOURCE, n.relationships)
        self.assertIn(NodeRelationship.PREVIOUS, n.relationships)
        self.assertIn(NodeRelationship.NEXT, n.relationships)
        self.assertEqual(SAMPLE[nodes[0].start_char_idx:nodes[0].end_char_idx], nodes[0].text)
        embed_text = n.get_content(metadata_mode=MetadataMode.EMBED)
        self.assertIn("headings:", embed_text)
        self.assertNotIn("start_index", embed_text)

    def test_semantic_with_llamaindex_embedding(self):
        nodes = self.Parser("semantic", embed_model=self.ToyEmbedding(), buffer=1, min_sentences=1) \
            .get_nodes_from_documents([self.Document(text=PROSE)])
        self.assertEqual(len(nodes), 2)

    def test_agentic_with_llamaindex_llm(self):
        nodes = self.Parser("agentic", llm=self.LLM()).get_nodes_from_documents([self.Document(text=PROSE)])
        self.assertEqual([n.metadata["title"] for n in nodes], ["Cats", "Rust"])

    def test_propositional_keeps_passage_for_llm_only(self):
        from llama_index.core.schema import MetadataMode
        nodes = self.Parser("propositional").get_nodes_from_documents(
            [self.Document(text="Redis is fast. It stores data in RAM.")])
        self.assertEqual(nodes[1].text, "Redis stores data in RAM.")
        self.assertNotIn("source_passage", nodes[1].get_content(metadata_mode=MetadataMode.EMBED))
        self.assertIn("source_passage", nodes[1].get_content(metadata_mode=MetadataMode.LLM))

    def test_chunking_embedding_in_index(self):
        from llama_index.core import VectorStoreIndex
        idx = VectorStoreIndex.from_documents([self.Document(text=SAMPLE)], embed_model=self.CE("hashing"),
                                              transformations=[self.Parser("document", max_tokens=80)])
        hit = idx.as_retriever(similarity_top_k=1).retrieve("spending cap for each team at the offsite")[0]
        self.assertIn("Team Notes", hit.node.metadata["headings"])


if __name__ == "__main__":
    unittest.main()
