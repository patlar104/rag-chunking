"""LLM backends. An LLM is any `Callable[[str], str]` (prompt -> reply text).

The classes here additionally implement `.complete_json(prompt, schema)`, which
asks the model for structured JSON (Ollama structured outputs / OpenAI JSON
mode), so small local models produce parseable answers far more reliably.

    get_llm("ollama:llama3.2:3b")    # local, via Ollama
    get_llm("openai:gpt-4o-mini")    # API
    get_llm("auto")                  # local Ollama chat model if one is installed, else OpenAI, else None
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

from . import events, ollama

LLM = Callable[[str], str]

_THINK = re.compile(r"<think>.*?</think>", re.S)


class OllamaLLM:
    """Chat model served by Ollama. Small models work well for chunking decisions:

        ollama pull llama3.2:3b     # 2.0 GB, good default
        ollama pull qwen2.5:1.5b    # 1.0 GB, fast on a laptop CPU
        ollama pull llama3.2:1b     # 1.3 GB
    """

    def __init__(self, model: str = "llama3.2:3b", host: str | None = None,
                 temperature: float = 0.0, num_ctx: int | None = 4096, keep_alive: str = "10m",
                 timeout: float = 300, system: str | None = None, think: bool | None = None):
        self.model, self.name = model, f"ollama:{model}"
        self.client = ollama.Client(host, timeout=timeout)
        self.options = {"temperature": temperature}
        if num_ctx:
            self.options["num_ctx"] = num_ctx
        self.keep_alive, self.system, self.think = keep_alive, system, think
        self._schema_ok = True
        self.calls = 0

    def _chat(self, prompt: str, fmt: Any = None) -> str:
        msgs = ([{"role": "system", "content": self.system}] if self.system else []) + \
               [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "stream": False,
                "options": self.options, "keep_alive": self.keep_alive}
        if fmt is not None:
            body["format"] = fmt
        if self.think is not None:
            body["think"] = self.think
        self.calls += 1
        with events.timed("llm.call", backend=self.name, json=fmt is not None,
                          prompt_chars=len(prompt)) as info:
            r = self.client.post("/api/chat", body)
            info.update(_ollama_usage(r))
        return _THINK.sub("", r["message"]["content"]).strip()

    def __call__(self, prompt: str) -> str:
        return self._chat(prompt)

    def complete_json(self, prompt: str, schema: dict | None = None) -> str:
        if schema and self._schema_ok:
            try:
                return self._chat(prompt, schema)          # structured outputs (Ollama >= 0.5)
            except ollama.OllamaError as e:
                if isinstance(e, (ollama.OllamaNotRunning, ollama.ModelNotFound)):
                    raise
                self._schema_ok = False                    # older server: plain JSON mode
        return self._chat(prompt, "json")

    def __repr__(self):
        return f"<OllamaLLM {self.model}>"


class OpenAILLM:
    """`pip install openai`, set OPENAI_API_KEY. Pass `base_url` for any
    OpenAI-compatible server (LM Studio, vLLM, llama.cpp, Groq, ...)."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 base_url: str | None = None, api_key: str | None = None):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("openai is not installed. Run:\n  pip install openai") from e
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.name, self.temperature = model, f"openai:{model}", temperature
        self.calls = 0

    def _chat(self, prompt: str, **kw) -> str:
        self.calls += 1
        with events.timed("llm.call", backend=self.name, json=bool(kw), prompt_chars=len(prompt)) as info:
            r = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}], **kw)
            if getattr(r, "usage", None):
                info.update(prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens)
        return r.choices[0].message.content or ""

    def __call__(self, prompt: str) -> str:
        return self._chat(prompt)

    def complete_json(self, prompt: str, schema: dict | None = None) -> str:
        return self._chat(prompt, response_format={"type": "json_object"})


def _ollama_usage(r: dict) -> dict:
    """Token counts and speed from an Ollama response (durations are in ns)."""
    out = {}
    if "prompt_eval_count" in r:
        out["prompt_tokens"] = r["prompt_eval_count"]
    if "eval_count" in r:
        out["completion_tokens"] = r["eval_count"]
        if r.get("eval_duration"):
            out["tokens_per_s"] = round(r["eval_count"] / (r["eval_duration"] / 1e9), 1)
    if r.get("load_duration"):
        out["load_ms"] = round(r["load_duration"] / 1e6, 1)
    return out


def get_llm(spec: str | None, **kwargs):
    """'ollama[:model]', 'openai[:model]', 'deepinfra[:model]', 'auto', or None/'none'."""
    if not spec or spec == "none":
        return None
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return OllamaLLM(model or "llama3.2:3b", **kwargs)
    if kind == "openai":
        return OpenAILLM(model or "gpt-4o-mini", **kwargs)
    if kind == "deepinfra":
        from .deepinfra import DEFAULT_LLM, DeepInfraLLM
        return DeepInfraLLM(model or DEFAULT_LLM, **kwargs)
    if kind == "auto":
        client = ollama.Client(kwargs.get("host"), timeout=3)
        if client.is_running():
            chat = [m for m in client.list_models() if not _looks_like_embedder(m)]
            if chat:
                return OllamaLLM(chat[0], **kwargs)
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAILLM(**kwargs)
        if os.environ.get("DEEPINFRA_API_KEY"):
            from .deepinfra import DeepInfraLLM
            return DeepInfraLLM(**kwargs)
        return None
    raise ValueError(f"unknown LLM {spec!r}; use ollama[:model], openai[:model], deepinfra[:model], auto or none")


def _looks_like_embedder(name: str) -> bool:
    return any(t in name for t in ("embed", "minilm", "bge", "e5-", "gte", "arctic"))


def ask_json(llm, prompt: str, schema: dict | None = None, retries: int = 1) -> Any:
    """Ask for JSON, using the backend's JSON mode when it has one; retry on bad output."""
    last_err = None
    for attempt in range(retries + 1):
        p = prompt if attempt == 0 else prompt + "\n\nYour last reply was not valid JSON. Reply with JSON only."
        complete = getattr(llm, "complete_json", None)
        reply = complete(p, schema) if complete else llm(p)
        try:
            return parse_json(reply)
        except ValueError as e:
            last_err = e
            if attempt < retries:
                events.emit("llm.retry", backend=getattr(llm, "name", "custom"), reason=str(e)[:200])
    events.emit("llm.json_failure", backend=getattr(llm, "name", "custom"), reason=str(last_err)[:200])
    if last_err is None:  # only possible with retries < 0
        raise ValueError("ask_json needs retries >= 0")
    raise last_err


def parse_json(reply: str) -> Any:
    """Pull the first JSON object/array out of a model reply (handles ```json fences)."""
    reply = _THINK.sub("", reply)
    m = re.search(r"```(?:json)?\s*(.*?)```", reply, re.S)
    candidate = m.group(1) if m else reply
    start = min([i for i in (candidate.find("["), candidate.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"no JSON found in model reply: {reply[:200]!r}")
    try:
        return json.JSONDecoder().raw_decode(candidate[start:])[0]
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in model reply: {e}: {reply[:200]!r}") from None


# Backwards-compatible helpers from v1.
def openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> LLM:
    return OpenAILLM(model, temperature)


def ollama_llm(model: str = "llama3.2:3b", host: str | None = None) -> LLM:
    return OllamaLLM(model, host)
