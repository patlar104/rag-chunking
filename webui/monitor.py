"""Health, stability and state tracking for the web UI (framework-free, thread-safe)."""
from __future__ import annotations

import os
import sqlite3
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from chunking import events, ollama
from chunking.embeddings import get_embedder
from chunking.llm import get_llm

RANK = {"ok": 0, "off": 0, "degraded": 1, "down": 2}


def pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


class Metrics:
    """Rolling call metrics fed by chunking.events. Window: 10 minutes, 5 s buckets."""

    WINDOW, BUCKET = 600, 5

    def __init__(self):
        self.lock = threading.Lock()
        self.calls: Deque[Dict] = deque(maxlen=20000)
        self.marks: Deque[Dict] = deque(maxlen=5000)   # retries, failures, fallbacks
        self.cache = {"hits": 0, "misses": 0}
        self.totals = {"embed": 0, "llm": 0, "errors": 0, "retries": 0, "json_failures": 0, "fallbacks": 0}

    def record(self, ev: Dict) -> None:
        t = ev["type"]
        with self.lock:
            if t in ("embed.call", "llm.call"):
                op = t.split(".")[0]
                self.calls.append({"ts": ev["ts"], "op": op, "backend": ev.get("backend", "?"),
                                   "ms": ev.get("ms", 0.0), "ok": ev.get("ok", True), "error": ev.get("error"),
                                   "tps": ev.get("tokens_per_s"), "probe": ev.get("run") == "probe"})
                self.totals[op] += 1
                if not ev.get("ok", True):
                    self.totals["errors"] += 1
            elif t == "cache.lookup":
                self.cache["hits"] += ev.get("hits", 0)
                self.cache["misses"] += ev.get("misses", 0)
            elif t in ("llm.retry", "llm.json_failure", "fallback"):
                key = {"llm.retry": "retries", "llm.json_failure": "json_failures", "fallback": "fallbacks"}[t]
                self.totals[key] += 1
                self.marks.append({"ts": ev["ts"], "kind": key})

    def _op_summary(self, rows: List[Dict], now: float) -> Dict[str, Any]:
        recent = [r for r in rows if now - r["ts"] <= 300]
        ms = [r["ms"] for r in recent if r["ok"]]
        errs = [r for r in recent if not r["ok"]]
        mean = statistics.fmean(ms) if ms else None
        cv = (statistics.pstdev(ms) / mean) if ms and len(ms) > 1 and mean else None
        tps = [r["tps"] for r in recent if r.get("tps")]
        last_err = next((r for r in reversed(rows) if not r["ok"]), None)
        return {
            "calls_5m": len(recent), "errors_5m": len(errs),
            "error_rate_5m": round(len(errs) / len(recent), 4) if recent else 0.0,
            "p50": pct(ms, 50), "p95": pct(ms, 95), "p99": pct(ms, 99),
            "mean": round(mean, 2) if mean else None,
            "jitter_cv": round(cv, 3) if cv is not None else None,
            "tokens_per_s": round(statistics.fmean(tps), 1) if tps else None,
            "last_ms": rows[-1]["ms"] if rows else None,
            "last_error": {"ts": last_err["ts"], "error": last_err["error"]} if last_err else None,
            "backends": sorted({r["backend"] for r in recent}),
        }

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self.lock:
            calls = [c for c in self.calls if now - c["ts"] <= self.WINDOW]
            marks = [m for m in self.marks if now - m["ts"] <= self.WINDOW]
            cache, totals = dict(self.cache), dict(self.totals)
        work = [c for c in calls if not c["probe"]]
        ops = {op: self._op_summary([c for c in work if c["op"] == op], now) for op in ("embed", "llm")}

        start = now - self.WINDOW
        n = self.WINDOW // self.BUCKET
        buckets = [{"t": round(start + i * self.BUCKET, 1), "embed": [], "llm": [], "err": 0, "calls": 0}
                   for i in range(n)]
        for c in work:
            i = min(n - 1, int((c["ts"] - start) // self.BUCKET))
            b = buckets[i]
            b["calls"] += 1
            if c["ok"]:
                b[c["op"]].append(c["ms"])
            else:
                b["err"] += 1
        series = [{"t": b["t"], "calls": b["calls"], "errors": b["err"],
                   "embed_p50": pct(b["embed"], 50), "embed_p95": pct(b["embed"], 95),
                   "llm_p50": pct(b["llm"], 50), "llm_p95": pct(b["llm"], 95)} for b in buckets]

        recent_marks = [m for m in marks if now - m["ts"] <= 300]
        llm_calls = ops["llm"]["calls_5m"]
        looked_up = cache["hits"] + cache["misses"]
        return {
            "ts": now, "ops": ops, "series": series, "totals": totals,
            "cache": dict(cache, hit_rate=round(cache["hits"] / looked_up, 4) if looked_up else None),
            "stability": {
                "retries_5m": sum(m["kind"] == "retries" for m in recent_marks),
                "fallbacks_5m": sum(m["kind"] == "fallbacks" for m in recent_marks),
                "json_failures_5m": sum(m["kind"] == "json_failures" for m in recent_marks),
                "retry_rate": round(sum(m["kind"] == "retries" for m in recent_marks) / llm_calls, 4) if llm_calls else None,
                "error_rate_5m": round(sum(o["errors_5m"] for o in ops.values()) /
                                       max(1, sum(o["calls_5m"] for o in ops.values())), 4),
            },
        }


class HealthMonitor:
    """Polls backends on a background thread and keeps a short history per component."""

    INTERVAL = 2.0
    EMBED_PROBE_EVERY = 15.0
    LLM_PROBE_EVERY = 60.0

    def __init__(self, metrics: Metrics, runner_state, host: Optional[str] = None):
        self.metrics, self.runner_state = metrics, runner_state
        self.host = host
        self.lock = threading.Lock()
        self.embed_spec, self.llm_spec = "hashing", "none"
        self._embed_probe = self._llm_probe = None
        self._last_embed_probe = self._last_llm_probe = 0.0
        self._probe_results: Dict[str, Dict] = {}
        self.history: Dict[str, Deque] = {k: deque(maxlen=150) for k in
                                          ("ollama", "embedder", "llm", "overall")}
        self.transitions: Deque = deque(maxlen=500)
        self.started = time.time()
        self._cpu = (time.process_time(), time.time())
        self.snapshot: Dict[str, Any] = {}
        self._force = threading.Event()
        self._stop = threading.Event()

    # -- configuration ---------------------------------------------------------------
    def configure(self, embed_spec: str, llm_spec: str):
        with self.lock:
            self.embed_spec, self.llm_spec = embed_spec, llm_spec
            self._embed_probe = self._llm_probe = None
            self._last_embed_probe = self._last_llm_probe = 0.0
            self._probe_results.clear()
        self.probe_now()

    def probe_now(self):
        with self.lock:
            self._last_embed_probe = self._last_llm_probe = 0.0
        self._force.set()

    def start(self, on_update):
        self.on_update = on_update
        threading.Thread(target=self._loop, name="health-monitor", daemon=True).start()

    def stop(self):
        self._stop.set()
        self._force.set()

    def _loop(self):
        token = events.current_run.set("probe")
        try:
            while not self._stop.is_set():
                try:
                    snap = self._collect()
                    self.snapshot = snap
                    self.on_update(snap)
                except Exception as e:  # never let the monitor die
                    self.snapshot = dict(self.snapshot, monitor_error=str(e))
                self._force.wait(self.INTERVAL)
                self._force.clear()
        finally:
            events.current_run.reset(token)

    # -- probes -----------------------------------------------------------------------
    def _uses_ollama(self) -> bool:
        return self.embed_spec.startswith(("ollama", "auto")) or self.llm_spec.startswith(("ollama", "auto"))

    def _ollama(self) -> Dict[str, Any]:
        client = ollama.Client(self.host, timeout=2)
        t0 = time.perf_counter()
        try:
            version = client.version()
        except ollama.OllamaError as e:
            return {"status": "down" if self._uses_ollama() else "off", "host": client.host,
                    "error": str(e).splitlines()[0], "required": self._uses_ollama()}
        ping = round((time.perf_counter() - t0) * 1000, 1)
        loaded, installed = [], []
        try:
            for m in client.get("/api/ps").get("models", []):
                loaded.append({"name": m.get("name"), "size": m.get("size"), "size_vram": m.get("size_vram", 0),
                               "expires_at": m.get("expires_at"), "context_length": m.get("context_length"),
                               "quant": (m.get("details") or {}).get("quantization_level"),
                               "params": (m.get("details") or {}).get("parameter_size")})
            installed = client.list_models()
        except ollama.OllamaError:
            pass  # the ping worked; model lists are best-effort and stay empty
        status = "ok" if ping < 250 else "degraded"
        return {"status": status, "host": client.host, "version": version, "ping_ms": ping,
                "loaded": loaded, "installed": installed, "required": self._uses_ollama()}

    def _probe_embedder(self, now: float) -> Dict[str, Any]:
        spec = self.embed_spec
        if now - self._last_embed_probe < self.EMBED_PROBE_EVERY and "embedder" in self._probe_results:
            return self._probe_results["embedder"]
        self._last_embed_probe = now
        res: Dict[str, Any] = {"spec": spec}
        try:
            if self._embed_probe is None:
                self._embed_probe = get_embedder(spec, cache=False, **({"host": self.host} if self.host and spec.startswith(("ollama", "auto")) else {}))
            t0 = time.perf_counter()
            vec = self._embed_probe([f"health probe {int(now)}"])[0]
            ms = round((time.perf_counter() - t0) * 1000, 1)
            res.update(name=getattr(self._embed_probe, "name", spec), dim=len(vec), probe_ms=ms,
                       status="ok" if ms < 2000 else "degraded", probed_at=now)
        except Exception as e:
            res.update(status="down", error=str(e).splitlines()[0], probed_at=now)
            self._embed_probe = None
        self._probe_results["embedder"] = res
        return res

    def _probe_llm(self, now: float) -> Dict[str, Any]:
        spec = self.llm_spec
        if spec in ("none", ""):
            return {"spec": "none", "status": "off", "name": "offline heuristics"}
        if now - self._last_llm_probe < self.LLM_PROBE_EVERY and "llm" in self._probe_results:
            return self._probe_results["llm"]
        self._last_llm_probe = now
        res: Dict[str, Any] = {"spec": spec}
        try:
            if self._llm_probe is None:
                self._llm_probe = get_llm(spec, **({"host": self.host} if self.host and spec.startswith(("ollama", "auto")) else {}))
                if self._llm_probe is None:
                    raise RuntimeError("no LLM available for 'auto'")
                if hasattr(self._llm_probe, "options"):
                    self._llm_probe.options["num_predict"] = 3
            t0 = time.perf_counter()
            reply = self._llm_probe("Reply with the single word OK.")
            ms = round((time.perf_counter() - t0) * 1000, 1)
            res.update(name=getattr(self._llm_probe, "name", spec), probe_ms=ms, reply=reply[:20],
                       status="ok" if ms < 8000 else "degraded", probed_at=now)
        except Exception as e:
            res.update(status="down", error=str(e).splitlines()[0], probed_at=now)
            self._llm_probe = None
        self._probe_results["llm"] = res
        return res

    def _hosted(self, emb, llm, m) -> List[Dict[str, Any]]:
        """Hosted (API) models in use, for the Model state panel."""
        out = []
        for kind, spec, comp, op in (("embedding", self.embed_spec, emb, "embed"), ("chat", self.llm_spec, llm, "llm")):
            provider, _, model = spec.partition(":")
            if provider not in ("deepinfra", "openai"):
                continue
            o = m["ops"][op]
            out.append({"kind": kind, "provider": {"deepinfra": "DeepInfra", "openai": "OpenAI"}[provider],
                        "model": model or comp.get("name", spec), "status": comp.get("status"),
                        "dim": comp.get("dim"), "probe_ms": comp.get("probe_ms"), "p95": o["p95"],
                        "calls_5m": o["calls_5m"], "tokens_per_s": o.get("tokens_per_s")})
        return out

    def _cache(self) -> Dict[str, Any]:
        d = Path(os.environ.get("CHUNKING_CACHE_DIR", Path.home() / ".cache" / "rag-chunking"))
        path = d / "embeddings.sqlite"
        info: Dict[str, Any] = {"path": str(path), "status": "ok"}
        if path.exists():
            info["bytes"] = path.stat().st_size
            try:
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
                info["entries"] = con.execute("SELECT count(*) FROM emb").fetchone()[0]
                con.close()
            except sqlite3.Error as e:
                info.update(status="degraded", error=str(e))
        else:
            info.update(bytes=0, entries=0)
        return info

    def _process(self, now: float) -> Dict[str, Any]:
        rss = None
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) / 1024
        except OSError:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if os.uname().sysname == "Linux" else 1024 * 1024)
        cpu_t, wall = time.process_time(), now
        prev_cpu, prev_wall = self._cpu
        self._cpu = (cpu_t, wall)
        cpu_pct = 100 * (cpu_t - prev_cpu) / max(1e-6, wall - prev_wall)
        load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
        return {"status": "ok", "rss_mb": round(rss, 1) if rss else None, "cpu_pct": round(cpu_pct, 1),
                "threads": threading.active_count(), "load": [round(x, 2) if x is not None else None for x in load],
                "cpus": os.cpu_count(), "uptime_s": round(now - self.started)}

    # -- aggregation ------------------------------------------------------------------------
    def _collect(self) -> Dict[str, Any]:
        now = time.time()
        worker = self.runner_state()
        busy = worker.get("state") == "running"
        with self.lock:
            oll = self._ollama()
            if busy and "embedder" in self._probe_results:
                # don't compete with the job for CPU; live call metrics cover this period
                emb = dict(self._probe_results["embedder"], paused=True)
                self._last_embed_probe = now
            else:
                emb = self._probe_embedder(now)
            if busy and "llm" in self._probe_results and self.llm_spec not in ("none", ""):
                llm = dict(self._probe_results["llm"], paused=True)
                self._last_llm_probe = now - self.LLM_PROBE_EVERY + 5  # re-probe soon after the run
            else:
                llm = self._probe_llm(now)
        m = self.metrics.snapshot()

        # backends that live in Ollama go down with it, and are re-probed as soon as it's back
        oll_down = oll.get("status") == "down"
        if oll_down:
            if self.embed_spec.startswith("ollama"):
                emb = dict(emb, status="down", error="Ollama unreachable")
            if self.llm_spec.startswith("ollama"):
                llm = dict(llm, status="down", error="Ollama unreachable")
        elif getattr(self, "_oll_was_down", False):
            self.probe_now()
        self._oll_was_down = oll_down

        # fold live error/latency rates into component status
        e_ops, l_ops = m["ops"]["embed"], m["ops"]["llm"]
        if emb.get("status") == "ok" and e_ops["calls_5m"] >= 5 and e_ops["error_rate_5m"] > 0.05:
            emb = dict(emb, status="degraded", reason=f"{e_ops['error_rate_5m']:.0%} errors (5m)")
        if llm.get("status") == "ok" and l_ops["calls_5m"] >= 5:
            fb = m["stability"]["fallbacks_5m"] + m["stability"]["json_failures_5m"]
            if l_ops["error_rate_5m"] > 0.05:
                llm = dict(llm, status="degraded", reason=f"{l_ops['error_rate_5m']:.0%} errors (5m)")
            elif fb / l_ops["calls_5m"] > 0.1:
                llm = dict(llm, status="degraded", reason=f"{fb} fallbacks (5m)")
            elif (l_ops["p95"] or 0) > 20000:
                llm = dict(llm, status="degraded", reason="p95 latency > 20 s")

        cache = self._cache()
        proc = self._process(now)
        if not worker.get("alive", True):
            worker["status"] = "down"
        components = {"ollama": oll, "embedder": emb, "llm": llm, "cache": cache, "worker": worker, "process": proc,
                      "hosted": self._hosted(emb, llm, m)}
        required = ["embedder", "worker"] + (["ollama"] if oll.get("required") else []) + \
                   (["llm"] if llm.get("status") != "off" else [])
        overall = max((components[k].get("status", "ok") for k in required), key=lambda s: RANK.get(s, 0))

        prev = self.history["overall"][-1]["s"] if self.history["overall"] else None
        if prev is not None and prev != overall:
            self.transitions.append({"ts": now, "from": prev, "to": overall})
        for key, comp in (("ollama", oll), ("embedder", emb), ("llm", llm)):
            lat = comp.get("ping_ms") if key == "ollama" else comp.get("probe_ms")
            self.history[key].append({"t": now, "s": comp.get("status"), "ms": lat})
        self.history["overall"].append({"t": now, "s": overall})

        window = [h for h in self.history["overall"] if now - h["t"] <= 300]
        ok_share = sum(h["s"] in ("ok", "off") for h in window) / len(window) if window else 1.0
        flaps = sum(1 for t in self.transitions if now - t["ts"] <= 600)
        return {
            "ts": now, "overall": overall, "components": components,
            "history": {k: list(v) for k, v in self.history.items()},
            "stability": {"healthy_share_5m": round(ok_share, 4), "status_changes_10m": flaps,
                          "transitions": list(self.transitions)[-20:]},
            "settings": {"embed": self.embed_spec, "llm": self.llm_spec},
        }
