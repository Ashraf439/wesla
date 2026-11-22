"""`python -m wesla doctor` - checks every dependency the live app needs, one at a
time, so a crash or hang points at a single component. Results are also logged."""
from __future__ import annotations

import importlib
import platform
import sys
import time

from .log import get_logger

log = get_logger("doctor")


def _check(name: str, fn) -> bool:
    t0 = time.perf_counter()
    print(f"[....] {name}", end="", flush=True)
    log.info("doctor: checking %s", name)
    try:
        detail = fn()
        dt = time.perf_counter() - t0
        print(f"\r[ OK ] {name} ({dt:.1f}s){' - ' + detail if detail else ''}")
        log.info("doctor: OK %s (%.1fs) %s", name, dt, detail or "")
        return True
    except Exception as exc:
        dt = time.perf_counter() - t0
        print(f"\r[FAIL] {name} ({dt:.1f}s) - {type(exc).__name__}: {exc}")
        log.exception("doctor: FAIL %s", name)
        return False


def run_doctor(cfg, quick: bool = False) -> int:
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()} ({sys.executable})")
    if ".venv" not in sys.executable and "venv" not in sys.prefix.lower():
        print("[WARN] this does not look like the project venv - activate it with .venv\\Scripts\\Activate.ps1")
    ok = True

    def ver(mod):
        def f():
            m = importlib.import_module(mod)
            return getattr(m, "__version__", "installed")
        return f

    for mod in ("numpy", "requests", "sentence_transformers", "gradio", "ddgs", "trafilatura", "datasketch"):
        ok &= _check(f"import {mod}", ver(mod))

    def torch_info():
        import torch
        cuda = torch.cuda.is_available()
        return f"{torch.__version__}, CUDA {'yes - ' + torch.cuda.get_device_name(0) if cuda else 'no (CPU build or no GPU)'}"
    ok &= _check("import torch", torch_info)

    def faiss_info():
        from . import vector_index
        try:
            import faiss  # noqa: F401
            installed = True
        except Exception:
            installed = False
        using = vector_index.faiss is not None
        msg = f"installed={installed}, used={using}"
        if installed and not using:
            msg += " (NumPy index used to avoid the Windows OpenMP crash; set WESLA_USE_FAISS=1 to force)"
        return msg
    _check("vector index backend", faiss_info)

    def ollama():
        import requests
        base = cfg.llm.ollama_url.rstrip("/")
        v = requests.get(f"{base}/api/version", timeout=5).json().get("version", "?")
        tags = [m["name"] for m in requests.get(f"{base}/api/tags", timeout=5).json().get("models", [])]
        want = cfg.llm.ollama_model
        have = any(t == want or t.split(":")[0] == want.split(":")[0] and want in t for t in tags) or want in tags
        if not have:
            raise RuntimeError(f"Ollama {v} is running but model {want!r} is missing - run: ollama pull {want}. "
                               f"Installed: {tags}")
        return f"Ollama {v}, model {want} present"
    if "ollama" in cfg.llm.providers:
        ok &= _check("Ollama server + base model", ollama)

    def embedder():
        from .embeddings import make_embedder
        e = make_embedder(cfg.embedding.backend, cfg.embedding.model, cfg.embedding.dim)
        v = e.embed(["hello world", "a test sentence"])
        return f"{type(e).__name__}, dim {v.shape[1]}"
    ok &= _check(f"embedding model ({cfg.embedding.model})", embedder)

    def faiss_torch_together():
        # reproduces the Windows OpenMP clash: torch embedding then a vector index search
        import numpy as np
        from .embeddings import make_embedder
        from .vector_index import VectorIndex
        e = make_embedder(cfg.embedding.backend, cfg.embedding.model, cfg.embedding.dim)
        idx = VectorIndex(e.dim)
        idx.add([1, 2], e.embed(["alpha", "beta"]))
        hits = idx.search(e.embed(["alpha"])[0], k=1)
        assert hits and hits[0][0] == 1, hits
        return "embedding + index search work together"
    ok &= _check("embedder + vector index in one process", faiss_torch_together)

    if cfg.verify.use_nli:
        def nli():
            from .verify import CrossEncoderNLI
            m = CrossEncoderNLI(cfg.verify.nli_model)
            p = m.contradiction("The sky is blue.", "The sky is not blue.")
            return f"contradiction prob on a negation pair = {p:.2f}"
        ok &= _check(f"NLI model ({cfg.verify.nli_model})", nli)

    if not quick:
        def search():
            from .search import make_search
            s = make_search(cfg.search.providers, max_retries=1, backoff_s=0)
            res = s.search("retrieval augmented generation", 3)
            if not res:
                raise RuntimeError(f"no results; errors: {s.last_errors}")
            return f"{len(res)} result(s), first: {res[0].url}"
        ok &= _check("web search", search)

        def fetch():
            from .extract import fetch_text
            text, _ = fetch_text("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
            if not text:
                raise RuntimeError("page fetch returned nothing (trafilatura and Jina Reader both failed)")
            return f"{len(text)} chars extracted"
        ok &= _check("page fetch + extraction", fetch)

        def llm():
            from .llm import make_llm
            chain = make_llm(cfg.llm)
            g = chain.chat([{"role": "user", "content": "Reply with the single word: ready"}],
                           max_tokens=8, want_logprobs=True)
            ent = "no logprobs (entropy route disabled - update Ollama)" if g.mean_entropy is None \
                else f"logprobs OK (entropy {g.mean_entropy:.2f})"
            return f"{g.provider}:{g.model} said {g.text.strip()[:30]!r}; {ent}"
        ok &= _check("LLM generation", llm)

    print("\nAll checks passed." if ok else "\nSome checks failed - see above and data/logs/wesla.log.")
    return 0 if ok else 1
