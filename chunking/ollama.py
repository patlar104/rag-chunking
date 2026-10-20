"""Minimal Ollama HTTP client (stdlib only).

Host resolution: explicit `host` argument > $OLLAMA_HOST > http://localhost:11434
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Iterator, List, Optional

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


class OllamaNotRunning(OllamaError):
    pass


class ModelNotFound(OllamaError):
    pass


class EndpointMissing(OllamaError):
    """The server is too old to have this endpoint."""


def resolve_host(host: Optional[str] = None) -> str:
    host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    if "://" not in host:
        host = "http://" + host
    host = host.rstrip("/")
    # OLLAMA_HOST=0.0.0.0 means "listen everywhere" on the server; connect locally.
    host = host.replace("://0.0.0.0", "://127.0.0.1")
    if host.count(":") == 1:  # no port given
        host += ":11434"
    return host


class Client:
    def __init__(self, host: Optional[str] = None, timeout: float = 120):
        self.host = resolve_host(host)
        self.timeout = timeout

    # -- transport -------------------------------------------------------------
    def _request(self, method: str, path: str, body: Optional[dict] = None, timeout=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.host + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            e.close()
            try:
                msg = json.loads(raw).get("error", raw)
            except ValueError:
                msg = raw
            model = (body or {}).get("model", "")
            if e.code == 404 and "not found" in msg and model and "page not found" not in msg:
                raise ModelNotFound(f"Ollama model {model!r} is not installed. Run:\n"
                                    f"  ollama pull {model}") from None
            if e.code == 404:
                raise EndpointMissing(f"{path} not available on this Ollama server ({msg.strip()}). "
                                      "Try updating Ollama.") from None
            if e.code == 400 and "does not support" in msg:
                raise OllamaError(f"{model!r} can't do this: {msg}. "
                                  "(Embedding models can't chat and chat models can't embed, "
                                  "pick the right kind.)") from None
            raise OllamaError(f"Ollama {e.code} on {path}: {msg}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            raise OllamaNotRunning(
                f"Can't reach Ollama at {self.host} ({reason}).\n"
                "  Install: https://ollama.com/download\n"
                "  Start:   ollama serve\n"
                "  Or set OLLAMA_HOST if it runs elsewhere.") from None

    def post(self, path: str, body: dict) -> dict:
        with self._request("POST", path, body) as r:
            return json.loads(r.read())

    def get(self, path: str, timeout=None) -> dict:
        with self._request("GET", path, timeout=timeout) as r:
            return json.loads(r.read())

    def stream(self, path: str, body: dict) -> Iterator[dict]:
        with self._request("POST", path, dict(body, stream=True)) as r:
            for line in r:
                if line.strip():
                    yield json.loads(line)

    # -- helpers -----------------------------------------------------------------
    def is_running(self) -> bool:
        try:
            self.get("/api/version", timeout=min(self.timeout, 3))
            return True
        except OllamaError:
            return False

    def version(self) -> str:
        return self.get("/api/version").get("version", "?")

    def list_models(self) -> List[str]:
        return [m["name"] for m in self.get("/api/tags").get("models", [])]

    def has_model(self, model: str) -> bool:
        names = self.list_models()
        return model in names or f"{model}:latest" in names

    def pull(self, model: str, quiet: bool = False) -> None:
        last = ""
        for ev in self.stream("/api/pull", {"model": model}):
            if "error" in ev:
                raise OllamaError(ev["error"])
            status = ev.get("status", "")
            if not quiet and ev.get("total"):
                pct = 100 * ev.get("completed", 0) / ev["total"]
                print(f"\r  {model}: {status} {pct:5.1f}%", end="", file=sys.stderr)
            elif not quiet and status != last:
                print(f"\r  {model}: {status}".ljust(60), file=sys.stderr)
            last = status
        if not quiet:
            print(file=sys.stderr)

    def ensure(self, model: str, pull: bool = False) -> None:
        if self.has_model(model):
            return
        if not pull:
            raise ModelNotFound(f"Ollama model {model!r} is not installed. Run:\n  ollama pull {model}")
        self.pull(model)
