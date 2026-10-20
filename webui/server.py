#!/usr/bin/env python3
"""Chunk Monitor: live web UI for rag-chunking.

    pip install fastapi uvicorn
    python -m webui.server              # http://localhost:8000
    python -m webui.server --port 8080 --embed ollama:nomic-embed-text --llm ollama:llama3.2:3b

Streams (Server-Sent Events at /api/stream):
    health   every 2 s: component status, probes, model state, process stats
    metrics  every 2 s: latency percentiles, error/retry/fallback rates, 10-minute series
    run.* / strategy.* / agentic.decision / semantic.distances / propositional.passage / call events
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from chunking import STRATEGIES, events, ollama  # noqa: E402
from chunking.embeddings import get_embedder  # noqa: E402
from chunking.llm import get_llm  # noqa: E402
from webui.monitor import HealthMonitor, Metrics  # noqa: E402
from webui.runner import Runner  # noqa: E402

STATIC = Path(__file__).with_name("static")
SAMPLES = {
    "redis-guide": ("Redis field guide (Markdown)", (ROOT / "sample.md").read_text(encoding="utf-8")),
    "topic-shifts": ("Topic shifts (plain prose)", (
        "Cats are small, carnivorous mammals that were domesticated thousands of years ago. They sleep for "
        "twelve to sixteen hours a day. Cats groom themselves with their rough tongues, which keeps their coat "
        "clean. A cat's whiskers help it judge whether it can fit through a gap.\n\n"
        "Rust is a systems programming language focused on safety and speed. It guarantees memory safety "
        "without a garbage collector. The borrow checker enforces ownership rules at compile time. Rust "
        "programs compile to native code and often match C in performance.\n\n"
        "The Great Barrier Reef is the world's largest coral reef system. It stretches for over two thousand "
        "kilometres off the coast of Queensland. Rising ocean temperatures have caused several mass bleaching "
        "events. Scientists monitor the reef with satellites and underwater surveys.\n\n"
        "Sourdough bread is leavened with a culture of wild yeast and lactic acid bacteria. The starter must be "
        "fed with flour and water regularly. Long fermentation gives the bread its tangy flavour. Bakers often "
        "score the loaf before baking so it can expand evenly.")),
}

# Event types forwarded to the browser (call events are trimmed to keep the stream light).
FORWARD = {"agentic.start", "agentic.decision", "semantic.distances", "propositional.passage",
           "llm.retry", "llm.json_failure", "fallback"}
CALLS = {"embed.call", "llm.call"}


class Hub:
    """Fan-out of SSE messages to every connected browser (thread-safe publish)."""

    def __init__(self):
        self.clients: set = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def publish(self, kind: str, data) -> None:
        if self.loop is None or not self.clients:
            return
        msg = f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
        try:
            self.loop.call_soon_threadsafe(self._fanout, msg)
        except RuntimeError:
            pass  # loop closed during shutdown

    def _fanout(self, msg: str):
        for q in list(self.clients):
            if q.qsize() < 2000:  # drop messages for a stuck client rather than grow forever
                q.put_nowait(msg)


hub = Hub()
metrics = Metrics()
runner = Runner(hub.publish)
monitor = HealthMonitor(metrics, runner.state, host=os.environ.get("OLLAMA_HOST"))
settings = {"embed": "hashing", "llm": "none"}


def _remember(ev):
    """Keep each run's live trace so a browser that connects mid-run can replay it."""
    run = runner.runs.get(ev.get("run"))
    if not run:
        return
    live, t = run.setdefault("live", {}), ev["type"]
    if t == "agentic.start":
        live["agentic"] = dict(live.get("agentic", {}), spans=ev["spans"], mode=ev["mode"], decisions=[])
    elif t == "agentic.decision":
        live.setdefault("agentic", {}).setdefault("decisions", []).append(ev)
    elif t == "semantic.distances":
        live.setdefault("semantic", {})["distances"] = ev
    elif t == "propositional.passage":
        live.setdefault("propositional", {}).setdefault("passages", []).append(ev)


def _count_call(ev):
    run = runner.runs.get(ev.get("run"))
    strategy = (run or {}).get("progress", {}) or {}
    if run and strategy.get("strategy") and not runner.in_trial:
        slot = run.setdefault("live", {}).setdefault(strategy["strategy"], {})
        slot["calls"] = slot.get("calls", 0) + 1


def on_event(ev):
    metrics.record(ev)
    t = ev["type"]
    if ev.get("run") == "probe":
        return
    if t in FORWARD:
        if not runner.in_trial:
            _remember(ev)
            hub.publish(t, ev)
    elif t in CALLS:
        _count_call(ev)
        hub.publish("call", {k: ev.get(k) for k in ("type", "ts", "run", "backend", "ms", "ok", "error", "n",
                                                     "tokens_per_s", "completion_tokens", "json")}
                    | {"trial": runner.in_trial})
    elif t == "cache.lookup":
        hub.publish("cache", ev)


def _metrics_pump():
    while True:
        time.sleep(2)
        hub.publish("metrics", metrics.snapshot())


@asynccontextmanager
async def lifespan(app):
    hub.loop = asyncio.get_running_loop()
    unsub = events.subscribe(on_event)
    monitor.configure(settings["embed"], settings["llm"])
    monitor.start(lambda snap: hub.publish("health", snap))
    threading.Thread(target=_metrics_pump, name="metrics-pump", daemon=True).start()
    yield
    monitor.stop()
    unsub()


app = FastAPI(title="Chunk Monitor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SettingsIn(BaseModel):
    embed: str = "hashing"
    llm: str = "none"


class RunIn(BaseModel):
    text: str = Field(..., max_length=200_000)
    strategies: List[str]
    size: int = Field(80, ge=10, le=2000)
    overlap: int = Field(0, ge=0, le=500)
    stability: bool = False


def _available():
    """What the user can pick in the UI, based on what's installed right now."""
    client = ollama.Client(monitor.host, timeout=2)
    embeds = [{"spec": "hashing", "label": "Hashing (offline toy)"}, {"spec": "auto", "label": "Auto-detect"}]
    llms = [{"spec": "none", "label": "None (offline heuristics)"}, {"spec": "auto", "label": "Auto-detect"}]
    running = client.is_running()
    if running:
        for m in client.list_models():
            name = m.removesuffix(":latest")
            is_embed = any(t in m for t in ("embed", "minilm", "bge", "e5", "gte", "arctic"))
            (embeds if is_embed else llms).append({"spec": f"ollama:{name}", "label": f"Ollama · {name}"})
    try:
        import sentence_transformers  # noqa: F401
        embeds.append({"spec": "st:all-MiniLM-L6-v2", "label": "sentence-transformers · all-MiniLM-L6-v2"})
    except ImportError:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        embeds.append({"spec": "openai:text-embedding-3-small", "label": "OpenAI · text-embedding-3-small"})
        llms.append({"spec": "openai:gpt-4o-mini", "label": "OpenAI · gpt-4o-mini"})
    return {"embed": embeds, "llm": llms, "ollama_running": running}


@app.get("/api/config")
def config():
    return {"strategies": list(STRATEGIES), "samples": {k: {"label": v[0], "text": v[1]} for k, v in SAMPLES.items()},
            "available": _available(), "settings": settings}


@app.post("/api/settings")
def update_settings(body: SettingsIn):
    try:  # validate before switching so a typo can't take the dashboard down
        get_embedder(body.embed)
        if body.llm not in ("none", "auto"):
            get_llm(body.llm)
    except Exception as e:
        raise HTTPException(400, str(e).splitlines()[0])
    settings.update(embed=body.embed, llm=body.llm)
    monitor.configure(body.embed, body.llm)
    hub.publish("settings", settings)
    return settings


@app.get("/api/health")
def health():
    return monitor.snapshot or {"overall": "unknown"}


@app.get("/api/metrics")
def get_metrics():
    return metrics.snapshot()


@app.post("/api/probe")
def probe():
    monitor.probe_now()
    return {"ok": True}


@app.post("/api/runs", status_code=201)
def create_run(body: RunIn):
    try:
        return runner.submit(body.text, body.strategies, body.size, body.overlap,
                             settings["embed"], settings["llm"], body.stability)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/runs")
def list_runs():
    return [runner.summary(runner.runs[i]) for i in reversed(runner.order) if i in runner.runs]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    d = runner.detail(run_id)
    if not d:
        raise HTTPException(404, "run not found")
    return d


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    if not runner.cancel(run_id):
        raise HTTPException(409, "run is not active")
    return {"ok": True}


@app.get("/api/stream")
async def stream(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    hub.clients.add(q)

    async def gen():
        try:
            hello = {"health": monitor.snapshot, "metrics": metrics.snapshot(), "worker": runner.state(),
                     "settings": settings}
            yield f"retry: 2000\nevent: hello\ndata: {json.dumps(hello, default=str)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    yield await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            hub.clients.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Static UI last, so /api/* routes win. Relative asset paths work locally and behind a proxy.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")


def main():
    ap = argparse.ArgumentParser(description="Chunk Monitor web UI")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to expose on your network")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--embed", default="hashing", help="initial embedder spec (changeable in the UI)")
    ap.add_argument("--llm", default="none", help="initial LLM spec (changeable in the UI)")
    args = ap.parse_args()
    settings.update(embed=args.embed, llm=args.llm)
    import uvicorn
    print(f"Chunk Monitor on http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
