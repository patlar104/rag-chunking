"""Background worker that runs chunking jobs and streams their progress."""
from __future__ import annotations

import itertools
import queue
import secrets
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

from chunking import STRATEGIES, chunk_stats, events, get_chunker, with_overlap
from chunking.embeddings import get_embedder
from chunking.llm import get_llm

LLM_STRATEGIES = {"agentic", "propositional"}


class Cancelled(Exception):
    pass


class BudgetExceeded(RuntimeError):
    pass


class _Guard:
    """Wraps an embedder/LLM so a cancelled run stops at the next model call,
    and (for LLMs) so the hourly spend budget is enforced."""

    def __init__(self, inner, run: dict, budget=None):
        self._inner, self._run, self._budget = inner, run, budget
        self.name = getattr(inner, "name", type(inner).__name__)

    def _check(self):
        if self._run["cancel"].is_set():
            raise Cancelled()
        if self._budget is not None and not self._budget():
            raise BudgetExceeded("hourly model-call budget for this public demo is used up; try again later")

    def __call__(self, *a, **k):
        self._check()
        return self._inner(*a, **k)

    def __getattr__(self, item):
        attr = getattr(self._inner, item)
        if callable(attr) and item in ("query", "complete_json"):
            def wrapped(*a, **k):
                self._check()
                return attr(*a, **k)
            return wrapped
        return attr


def boundaries(text: str, chunks) -> list[int]:
    real = [c for c in chunks if c.start >= 0]
    return sorted({c.end for c in real[:-1]})


def boundary_agreement(a: list[int], b: list[int], tol: int = 25) -> float:
    """F1 between two boundary sets; boundaries within `tol` chars count as the same cut."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    match_a = sum(any(abs(x - y) <= tol for y in b) for x in a)
    match_b = sum(any(abs(y - x) <= tol for x in a) for y in b)
    p, r = match_a / len(a), match_b / len(b)
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


class Runner:
    MAX_RUNS = 30

    def __init__(self, publish: Callable[[str, dict], None]):
        self.publish = publish
        self.queue: queue.Queue[str] = queue.Queue()
        self.runs: dict[str, dict] = {}
        self.order: list[str] = []
        self.ids = itertools.count(1)
        self.current: str | None = None
        self.completed = self.failed = 0
        self.in_trial = False  # stability re-runs: keep their events out of the live view
        self.llm_budget: int | None = None  # max LLM calls per rolling hour (None = unlimited)
        self._llm_calls: list[float] = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name="chunk-runner", daemon=True)
        self.thread.start()

    # -- public API ----------------------------------------------------------------------------
    def submit(self, text: str, strategies: list[str], size: int, overlap: int, embed: str, llm: str,
               stability: bool) -> dict:
        bad = [s for s in strategies if s not in STRATEGIES]
        if bad or not strategies:
            raise ValueError(f"unknown strategies: {bad or 'none selected'}")
        if not text.strip():
            raise ValueError("document is empty")
        run_id = f"r{next(self.ids)}"
        run = {"id": run_id, "status": "queued", "created": time.time(), "text": text,
               "strategies": strategies, "size": size, "overlap": overlap, "embed": embed, "llm": llm,
               "stability": stability, "results": {}, "error": None, "cancel": threading.Event(),
               "started": None, "finished": None, "cancel_token": secrets.token_urlsafe(16)}
        with self.lock:
            self.runs[run_id] = run
            self.order.append(run_id)
            while len(self.order) > self.MAX_RUNS:
                old = self.order.pop(0)
                if self.runs.get(old, {}).get("status") not in ("queued", "running"):
                    self.runs.pop(old, None)
        self.queue.put(run_id)
        self.publish("run.queued", self.summary(run))
        return self.summary(run)

    def cancel(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if not run or run["status"] not in ("queued", "running"):
            return False
        run["cancel"].set()
        if run["status"] == "queued":
            run["status"] = "cancelled"
            self.publish("run.finished", self.summary(run))
        return True

    def state(self) -> dict[str, Any]:
        cur = self.runs.get(self.current) if self.current else None
        queued = sum(1 for r in self.runs.values() if r["status"] == "queued")
        status = "ok" if queued <= 3 else "degraded"
        return {"status": status, "alive": self.thread.is_alive(), "llm_budget": self.budget_state(),
                "state": "running" if cur else "idle", "current": self.summary(cur) if cur else None,
                "queued": queued, "completed": self.completed, "failed": self.failed}

    def summary(self, run: dict | None) -> dict | None:
        if not run:
            return None
        return {k: run[k] for k in ("id", "status", "created", "started", "finished", "strategies", "size",
                                     "overlap", "embed", "llm", "stability", "error")} | {
            "progress": run.get("progress"),
            "done": [s for s, r in run["results"].items() if r.get("status") in ("done", "error")],
        }

    def detail(self, run_id: str) -> dict | None:
        run = self.runs.get(run_id)
        if not run:
            return None
        return self.summary(run) | {"text": run["text"], "results": run["results"], "live": run.get("live", {})}

    # -- worker ----------------------------------------------------------------------------------
    def _loop(self):
        while True:
            run_id = self.queue.get()
            run = self.runs.get(run_id)
            if not run or run["status"] == "cancelled":
                continue
            self.current = run_id
            token = events.current_run.set(run_id)
            try:
                self._execute(run)
            except Exception as e:  # pragma: no cover - defensive
                run.update(status="error", error=f"{type(e).__name__}: {e}")
                traceback.print_exc()
            finally:
                events.current_run.reset(token)
                self.current = None
                run["finished"] = time.time()
                if run["status"] == "error":
                    self.failed += 1
                elif run["status"] == "done":
                    self.completed += 1
                self.publish("run.finished", self.summary(run))

    def _backends(self, run):
        embed = llm = None
        embed = get_embedder(run["embed"], cache=True)
        if any(s in LLM_STRATEGIES for s in run["strategies"]):
            llm = get_llm(run["llm"])
        return _Guard(embed, run), (_Guard(llm, run, self._take_llm_call) if llm is not None else None)

    def _take_llm_call(self) -> bool:
        if self.llm_budget is None:
            return True
        now = time.time()
        with self.lock:
            self._llm_calls = [t for t in self._llm_calls if now - t < 3600]
            if len(self._llm_calls) >= self.llm_budget:
                return False
            self._llm_calls.append(now)
            return True

    def budget_state(self) -> dict[str, int] | None:
        if self.llm_budget is None:
            return None
        now = time.time()
        used = sum(1 for t in self._llm_calls if now - t < 3600)
        return {"used": used, "limit": self.llm_budget}

    def _build(self, name, size, overlap, embed, llm):
        if name == "fixed":
            return get_chunker("fixed", chunk_size=size, overlap=min(overlap, size - 1))
        if name == "recursive":
            return get_chunker("recursive", chunk_size=size)
        if name == "document":
            return get_chunker("document", max_tokens=size)
        if name == "semantic":
            return get_chunker("semantic", embed=embed, max_tokens=size * 2)
        if name == "propositional":
            return get_chunker("propositional", llm=llm)
        return get_chunker("agentic", llm=llm, max_tokens=size * 2)

    def _chunks(self, name, text, size, overlap, embed, llm):
        chunks = self._build(name, size, overlap, embed, llm).split(text)
        if overlap and name in ("recursive", "semantic"):
            chunks = with_overlap(text, chunks, overlap)
        return chunks

    def _execute(self, run):
        run.update(status="running", started=time.time())
        text = run["text"]
        self.publish("run.started", self.summary(run) | {"text": text})
        try:
            embed, llm = self._backends(run)
        except Exception as e:
            run.update(status="error", error=str(e).splitlines()[0])
            return
        n = len(run["strategies"])
        for i, name in enumerate(run["strategies"]):
            if run["cancel"].is_set():
                run["status"] = "cancelled"
                return
            run["progress"] = {"index": i, "total": n, "strategy": name}
            self.publish("strategy.started", {"run": run["id"], "strategy": name, "index": i, "total": n})
            t0 = time.perf_counter()
            res: dict[str, Any] = {"status": "running"}
            run["results"][name] = res
            try:
                chunks = self._chunks(name, text, run["size"], run["overlap"], embed, llm)
                secs = time.perf_counter() - t0
                res.update(status="done", seconds=round(secs, 6), stats=chunk_stats(text, chunks),
                           chunks=[{"start": c.start, "end": c.end, "tokens": c.tokens,
                                    "text": c.text if (c.start < 0 or text[c.start:c.end] != c.text) else None,
                                    "meta": {k: v for k, v in c.metadata.items()
                                             if k not in ("source_passage",) and isinstance(v, (str, int, float, bool, list, type(None)))}}
                                   for c in chunks])
                if run["stability"] and name != "propositional":
                    res["stability"] = self._stability(name, text, run, embed, llm, chunks)
            except Cancelled:
                res.update(status="cancelled")
                run["status"] = "cancelled"
                self.publish("strategy.finished", {"run": run["id"], "strategy": name, **res})
                return
            except Exception as e:
                res.update(status="error", error=f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}",
                           seconds=round(time.perf_counter() - t0, 3))
            self.publish("strategy.finished", {"run": run["id"], "strategy": name, **res})
        errors = [r for r in run["results"].values() if r.get("status") == "error"]
        run["status"] = "error" if errors and len(errors) == n else "done"
        if errors:
            run["error"] = f"{len(errors)} of {n} strategies failed"

    def _stability(self, name, text, run, embed, llm, base_chunks) -> dict[str, Any]:
        """Boundary stability: re-chunk with the size perturbed by ±15% and measure how many
        cut points survive (F1 with a 25-character tolerance)."""
        base = boundaries(text, base_chunks)
        trials = []
        for factor in (0.85, 1.15):
            size = max(5, int(run["size"] * factor))
            self.publish("stability.trial", {"run": run["id"], "strategy": name, "size": size})
            self.in_trial = True
            try:
                other = boundaries(text, self._chunks(name, text, size, run["overlap"], embed, llm))
            finally:
                self.in_trial = False
            trials.append({"size": size, "boundaries": len(other), "agreement": boundary_agreement(base, other)})
        score = round(sum(t["agreement"] for t in trials) / len(trials), 4)
        return {"score": score, "base_boundaries": len(base), "trials": trials}
