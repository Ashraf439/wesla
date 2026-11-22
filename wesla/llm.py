"""LLM providers behind one interface, with automatic fallback.

OllamaLLM   - the *learning* model (Llama 3.2 1B/3B + LoRA columns), local.
              Requests token logprobs so the router can compute predictive entropy.
OpenAICompatLLM - Groq / OpenRouter free tiers, used as a fallback generator or helper.
EchoLLM     - deterministic stub for tests and offline demos.
"""
from __future__ import annotations

import json
from .activity import checkpoint, current, Stopped
import math
import os
from dataclasses import dataclass
from typing import Protocol

import requests

from .log import get_logger

log = get_logger("llm")


@dataclass
class Generation:
    text: str
    provider: str
    model: str
    mean_entropy: float | None = None  # nats/token over top-k logprobs, if available


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], model: str | None = None, max_tokens: int = 512,
             temperature: float = 0.2, want_logprobs: bool = False) -> Generation: ...


def _entropy_from_top_logprobs(tokens: list[dict]) -> float | None:
    """tokens: [{"top_logprobs": [{"logprob": float}, ...]}, ...]"""
    ents = []
    for t in tokens:
        tops = t.get("top_logprobs") or []
        if not tops:
            continue
        ps = [math.exp(x["logprob"]) for x in tops]
        z = sum(ps)
        if z <= 0:
            continue
        ps = [p / z for p in ps]
        ents.append(-sum(p * math.log(p) for p in ps if p > 0))
    return sum(ents) / len(ents) if ents else None


class OllamaLLM:
    name = "ollama"

    def __init__(self, url: str, model: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, model=None, max_tokens=512, temperature=0.2, want_logprobs=False):
        body = {"model": model or self.model, "messages": messages, "stream": current.get() is not None,
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        if want_logprobs:
            body |= {"logprobs": True, "top_logprobs": 5}
        log.debug("ollama POST /api/chat model=%s logprobs=%s", body["model"], want_logprobs)
        try:
            r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout, stream=body["stream"])
        except requests.ConnectionError as exc:
            raise RuntimeError(f"cannot reach Ollama at {self.url} - is the Ollama app running? ({exc!s:.80})")
        except requests.Timeout:
            raise RuntimeError(f"Ollama did not answer within {self.timeout:.0f}s (model {body['model']}); "
                               f"raise [llm] timeout_s or use a smaller model")
        if r.status_code == 404 and body["model"] != self.model:
            log.warning("Ollama model %s not found; falling back to base model %s "
                        "(run `ollama create` for the column)", body["model"], self.model)
            body["model"] = self.model
            r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout, stream=body["stream"])
        if r.status_code == 404:
            raise RuntimeError(f"Ollama has no model {body['model']!r} - run: ollama pull {body['model']}")
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        if body["stream"]:
            parts, probs = [], []
            with r:
                for line in r.iter_lines(chunk_size=1):
                    checkpoint()
                    if line:
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise RuntimeError(chunk["error"])
                        parts.append(chunk.get("message", {}).get("content", ""))
                        probs.extend(chunk.get("logprobs") or [])
            data = {"message":{"content":"".join(parts)}, "logprobs":probs}
        else:
            data = r.json()
        ent = _entropy_from_top_logprobs(data.get("logprobs") or []) if want_logprobs else None
        return Generation(data["message"]["content"], self.name, body["model"], ent)

    def available_models(self) -> list[str]:
        r = requests.get(f"{self.url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


class OpenAICompatLLM:
    def __init__(self, name: str, base_url: str, model: str, api_key_env: str, timeout: float = 60.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout

    def chat(self, messages, model=None, max_tokens=512, temperature=0.2, want_logprobs=False):
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"{self.api_key_env} not set")
        r = requests.post(f"{self.base_url}/chat/completions", timeout=self.timeout,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": model or self.model, "messages": messages,
                                "max_tokens": max_tokens, "temperature": temperature})
        r.raise_for_status()
        data = r.json()
        return Generation(data["choices"][0]["message"]["content"], self.name, model or self.model)


class EchoLLM:
    """Answers by quoting the first context sentence; reports a configurable entropy."""
    name = "echo"

    def __init__(self, entropy: float | None = 2.0):
        self.entropy = entropy
        self.calls: list[list[dict]] = []

    def chat(self, messages, model=None, max_tokens=512, temperature=0.2, want_logprobs=False):
        self.calls.append(messages)
        user = messages[-1]["content"]
        if "CONTEXT:" in user:
            ctx = user.split("CONTEXT:", 1)[1].split("QUESTION:", 1)[0].strip()
            first = ctx.split("\n")[0].lstrip("[0123456789] ").strip()
            return Generation(f"Based on the sources: {first[:300]}", self.name, "echo",
                              self.entropy if want_logprobs else None)
        return Generation("I am not certain.", self.name, "echo",
                          self.entropy if want_logprobs else None)


class LLMChain:
    """Try providers in order; remember errors for the debug panel."""

    def __init__(self, providers: list[LLM]):
        if not providers:
            raise ValueError("No LLM providers configured")
        self.providers = providers
        self.last_errors: list[str] = []

    @property
    def primary(self) -> LLM:
        return self.providers[0]

    def chat(self, messages, model=None, **kw) -> Generation:
        self.last_errors = []
        for i, p in enumerate(self.providers):
            try:
                # a column-specific model name only applies to the local provider
                return p.chat(messages, model=model if i == 0 and p.name == "ollama" else None, **kw)
            except Stopped:
                raise
            except Exception as exc:
                log.warning("LLM provider %s failed: %s", p.name, exc)
                self.last_errors.append(f"{p.name}: {exc!s:.160}")
        raise RuntimeError("All LLM providers failed: " + " | ".join(self.last_errors))


def make_llm(cfg) -> LLMChain:
    out: list[LLM] = []
    for name in cfg.providers:
        if name == "ollama":
            out.append(OllamaLLM(cfg.ollama_url, cfg.ollama_model, cfg.timeout_s))
        elif name == "groq":
            out.append(OpenAICompatLLM("groq", "https://api.groq.com/openai/v1", cfg.groq_model,
                                       "GROQ_API_KEY", cfg.timeout_s))
        elif name == "openrouter":
            out.append(OpenAICompatLLM("openrouter", "https://openrouter.ai/api/v1",
                                       cfg.openrouter_model, "OPENROUTER_API_KEY", cfg.timeout_s))
        elif name == "echo":
            out.append(EchoLLM())
    return LLMChain(out)
