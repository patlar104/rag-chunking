"""Tiny in-process event bus used for telemetry and live visualisation.

Nothing is recorded unless something subscribes, so the overhead for normal
library use is a single list check per call.

    from chunking import events
    unsubscribe = events.subscribe(print)      # every event is a dict
    ...
    unsubscribe()

Event types emitted by the library:
    embed.call        backend, n, ms, ok, error?            (one per batch sent to a model)
    cache.lookup      backend, hits, misses
    llm.call          backend, ms, ok, error?, json, prompt_tokens?, completion_tokens?, tokens_per_s?
    llm.retry         backend, reason                        (bad JSON, asked again)
    llm.json_failure  backend, reason                        (gave up after retries)
    fallback          strategy, reason                       (LLM unusable -> offline rules)
    semantic.distances  gaps, distances, threshold, sentences
    agentic.start     sentences, spans, mode
    agentic.decision  index, start, end, boundary, reason, title, mode, ms
    propositional.passage  start, end, propositions, mode
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager

Subscriber = Callable[[dict], None]

_subscribers: list[Subscriber] = []
_lock = threading.Lock()

#: Set this (e.g. in a worker thread) to tag every event with a run id.
current_run: contextvars.ContextVar = contextvars.ContextVar("chunking_run", default=None)


def subscribe(fn: Subscriber) -> Callable[[], None]:
    with _lock:
        _subscribers.append(fn)

    def unsubscribe():
        with _lock:
            if fn in _subscribers:
                _subscribers.remove(fn)

    return unsubscribe


def active() -> bool:
    return bool(_subscribers)


def emit(kind: str, **data) -> None:
    if not _subscribers:
        return
    event = {"type": kind, "ts": time.time(), "run": current_run.get(), **data}
    for fn in list(_subscribers):
        try:
            fn(event)
        except Exception:  # a broken subscriber must never break chunking
            pass


@contextmanager
def timed(kind: str, **data):
    """Emit `kind` with duration (ms) and ok/error once the block finishes.
    The block can add fields by mutating the yielded dict."""
    extra: dict = {}
    t0 = time.perf_counter()
    try:
        yield extra
    except Exception as e:
        emit(
            kind,
            ms=round((time.perf_counter() - t0) * 1000, 2),
            ok=False,
            error=f"{type(e).__name__}: {e}"[:400],
            **data,
            **extra,
        )
        raise
    emit(kind, ms=round((time.perf_counter() - t0) * 1000, 2), ok=True, **data, **extra)
