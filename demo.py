#!/usr/bin/env python3
"""Run every chunking strategy on a document and compare them side by side.

    python demo.py                         # all strategies on sample.md
    python demo.py notes.md -s semantic    # one strategy on your own file
    python demo.py -s fixed --size 40 --overlap 10
    python demo.py --compare               # stats table only
    python demo.py -q "how big is the offsite budget?"   # top chunk per strategy

Real models (see `python demo.py --doctor` for what's available):
    python demo.py --embed ollama:nomic-embed-text --llm ollama:llama3.2:3b
    python demo.py --embed st:all-MiniLM-L6-v2
    python demo.py --embed openai --llm openai:gpt-4o-mini
    python demo.py --embed auto --llm auto     # pick the best local backends
    python demo.py --embed ollama:all-minilm --pull   # download missing Ollama models first
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

from chunking import STRATEGIES, chunk_stats, get_chunker, ollama, with_overlap
from chunking.embeddings import cosine, embed_query, get_embedder
from chunking.llm import get_llm

BAR = "─" * 78


def build(name: str, size: int, overlap: int, llm, embed):
    if name == "fixed":
        return get_chunker("fixed", chunk_size=size, overlap=overlap)
    if name == "recursive":
        return get_chunker("recursive", chunk_size=size)
    if name == "document":
        return get_chunker("document", max_tokens=size)
    if name == "semantic":
        return get_chunker("semantic", embed=embed, max_tokens=size * 2)
    if name == "propositional":
        return get_chunker("propositional", llm=llm)
    return get_chunker("agentic", llm=llm, max_tokens=size * 2)


def show(name, chunks, max_chunks, seconds):
    print(f"\n{BAR}\n{name.upper()}  ·  {len(chunks)} chunks  ·  {seconds:.2f}s\n{BAR}")
    for i, c in enumerate(chunks[:max_chunks], 1):
        md = {k: v for k, v in c.metadata.items() if k not in ("strategy", "source_passage") and v not in (None, [], "")}
        print(f"[{i}] {c.tokens} tok  {md}")
        print(textwrap.indent(textwrap.shorten(c.text.replace("\n", " ⏎ "), 300), "    "))
    if len(chunks) > max_chunks:
        print(f"    ... {len(chunks) - max_chunks} more")


def top_chunk(query, chunks, embed):
    q = embed_query(embed, query)
    vecs = embed([c.text for c in chunks])
    return max(((cosine(q, v), c) for v, c in zip(vecs, chunks)), key=lambda x: x[0])


def doctor(host):
    client = ollama.Client(host, timeout=3)
    print(f"Ollama ({client.host})")
    if client.is_running():
        models = client.list_models()
        print(f"  running, version {client.version()}")
        emb = [m for m in models if any(t in m for t in ("embed", "minilm", "bge", "e5", "gte", "arctic"))]
        chat = [m for m in models if m not in emb]
        print(f"  embedding models: {', '.join(emb) or 'none  ->  ollama pull nomic-embed-text'}")
        print(f"  chat models:      {', '.join(chat) or 'none  ->  ollama pull llama3.2:3b'}")
    else:
        print("  not reachable  ->  install from https://ollama.com/download, then `ollama serve`")
    try:
        import sentence_transformers
        print(f"sentence-transformers {sentence_transformers.__version__}: installed  (--embed st)")
    except ImportError:
        print("sentence-transformers: not installed  ->  pip install sentence-transformers")
    try:
        import openai
        key = "set" if os.environ.get("OPENAI_API_KEY") else "NOT set"
        print(f"openai {openai.__version__}: installed, OPENAI_API_KEY {key}")
    except ImportError:
        print("openai: not installed (optional)  ->  pip install openai")
    print("\n--embed auto picks:", end=" ", flush=True)
    print(getattr(get_embedder("auto", host=host), "name", "?"))
    llm = get_llm("auto", host=host)
    print("--llm auto picks:  ", getattr(llm, "name", "none (offline heuristics)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", default=str(Path(__file__).with_name("sample.md")))
    ap.add_argument("-s", "--strategy", choices=["all", *STRATEGIES], default="all")
    ap.add_argument("--size", type=int, default=80, help="target chunk size in tokens")
    ap.add_argument("--overlap", type=int, default=0, help="sliding-window overlap in tokens (try 20-50)")
    ap.add_argument("--max-show", type=int, default=8, help="chunks to print per strategy")
    ap.add_argument("--compare", action="store_true", help="print only the stats table")
    ap.add_argument("-q", "--query", help="retrieve the best chunk per strategy for this question")
    ap.add_argument("--embed", default="hashing",
                    help="hashing | auto | ollama[:model] | st[:model] | openai[:model]  (default: hashing)")
    ap.add_argument("--llm", default="none",
                    help="none | auto | ollama[:model] | openai[:model]  (for propositional + agentic)")
    ap.add_argument("--ollama-host", help="default: $OLLAMA_HOST or http://localhost:11434")
    ap.add_argument("--pull", action="store_true", help="pull missing Ollama models before running")
    ap.add_argument("--no-cache", action="store_true", help="don't cache embeddings on disk")
    ap.add_argument("--doctor", action="store_true", help="show which model backends are available")
    args = ap.parse_args()

    if args.doctor:
        return doctor(args.ollama_host)

    try:
        # Preflight: check Ollama models up front (and pull them with --pull).
        client = ollama.Client(args.ollama_host)
        for spec, default in ((args.embed, "nomic-embed-text"), (args.llm, "llama3.2:3b")):
            kind, _, model = spec.partition(":")
            if kind == "ollama":
                client.ensure(model or default, pull=args.pull)
        host_kw = {"host": args.ollama_host} if args.ollama_host else {}
        ekw = host_kw if args.embed.startswith(("ollama", "auto")) else {}
        lkw = host_kw if args.llm.startswith(("ollama", "auto")) else {}
        embed = get_embedder(args.embed, cache=not args.no_cache, **ekw)
        llm = get_llm(args.llm, **lkw)
    except (ollama.OllamaError, ImportError, ValueError) as e:
        sys.exit(f"error: {e}")
    except OSError as e:  # e.g. unknown Hugging Face model name
        sys.exit(f"error loading embedding model {args.embed!r}: {e}".splitlines()[0])

    print(f"embedder: {getattr(embed, 'name', embed)}   llm: {getattr(llm, 'name', 'none (offline heuristics)')}")
    text = Path(args.file).read_text(encoding="utf-8")
    names = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    rows, hits = [], []
    for name in names:
        t0 = time.perf_counter()
        try:
            chunks = build(name, args.size, args.overlap, llm, embed).split(text)
        except ollama.OllamaError as e:
            sys.exit(f"error in {name}: {e}")
        if args.overlap and name in ("recursive", "semantic", "document"):
            chunks = with_overlap(text, chunks, args.overlap) if name != "document" else chunks
        secs = time.perf_counter() - t0
        rows.append((name, chunk_stats(text, chunks), secs))
        if args.query:
            hits.append((name, *top_chunk(args.query, chunks, embed)))
        elif not args.compare:
            show(name, chunks, args.max_show, secs)

    if hits:
        print(f"\n{BAR}\nQUERY: {args.query}\n{BAR}")
        for name, score, chunk in hits:
            print(f"{name:<14} score {score:.3f}  ({chunk.tokens} tok)")
            print(textwrap.indent(textwrap.shorten(chunk.text.replace("\n", " ⏎ "), 240), "    "))
    else:
        print(f"\n{BAR}\nCOMPARISON  (target size {args.size} tokens, overlap {args.overlap})\n{BAR}")
        cols = ["chunks", "avg_tokens", "min_tokens", "max_tokens", "tiny_chunks", "mid_sentence_cuts"]
        print(f"{'strategy':<15}" + "".join(f"{c:>18}" for c in cols) + f"{'seconds':>10}")
        for name, st, secs in rows:
            print(f"{name:<15}" + "".join(f"{st.get(c, '-'):>18}" for c in cols) + f"{secs:>10.2f}")
        print("\nmid_sentence_cuts = boundaries that land inside a sentence (Problem #1: lost context).")
        print("propositional rewrites text, so its cut count isn't meaningful.")

    extras = []
    if hasattr(embed, "hits"):
        extras.append(f"embedding cache: {embed.hits} hits / {embed.misses} misses ({embed.path})")
    if getattr(llm, "calls", 0):
        extras.append(f"llm calls: {llm.calls}")
    if extras:
        print("\n" + "\n".join(extras))


if __name__ == "__main__":
    main()
