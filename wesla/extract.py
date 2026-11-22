"""Page fetching, boilerplate stripping and chunking (proposal §4.B).

Extraction order: trafilatura (best) -> Jina Reader (r.jina.ai) -> search snippet.
Chunks are 256-512 tokens on sentence boundaries with a small overlap.
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime

import time

from .log import get_logger
from .types import Chunk, SearchResult

log = get_logger("extract")

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def fetch_text(url: str, timeout: float = 20.0) -> tuple[str, datetime | None]:
    """Return (main_text, published_date). Empty string on failure."""
    t0 = time.perf_counter()
    try:
        import trafilatura

        html = trafilatura.fetch_url(url)
        if html:
            text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
            date = None
            try:
                meta = trafilatura.extract_metadata(html)
                if meta and meta.date:
                    date = datetime.fromisoformat(meta.date)
            except Exception:
                pass
            if text:
                log.debug("fetched %s via trafilatura: %d chars in %.2fs", url, len(text), time.perf_counter() - t0)
                return text, date
        log.debug("trafilatura got nothing for %s (%.2fs)", url, time.perf_counter() - t0)
    except Exception as exc:
        log.warning("trafilatura failed for %s: %s", url, exc)
    try:
        import requests

        r = requests.get(f"https://r.jina.ai/{url}", timeout=timeout,
                         headers={"X-Return-Format": "text"})
        if r.ok and len(r.text) > 200:
            log.debug("fetched %s via Jina Reader: %d chars", url, len(r.text))
            return r.text, None
        log.debug("Jina Reader HTTP %s for %s", r.status_code, url)
    except Exception as exc:
        log.warning("Jina Reader failed for %s: %s", url, exc)
    log.info("could not fetch %s after %.2fs; using search snippet", url, time.perf_counter() - t0)
    return "", None


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _ntok(s: str) -> int:
    return max(1, math.ceil(len(s.split()) / 0.75))


def chunk_text(text: str, min_tokens: int = 256, max_tokens: int = 512,
               overlap_tokens: int = 32) -> list[str]:
    sents = split_sentences(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for s in sents:
        t = _ntok(s)
        if cur and cur_tok + t > max_tokens:
            chunks.append(" ".join(cur))
            # carry over trailing sentences as overlap
            carry, ctok = [], 0
            for prev in reversed(cur):
                ctok += _ntok(prev)
                if ctok > overlap_tokens:
                    break
                carry.insert(0, prev)
            cur, cur_tok = carry, sum(_ntok(x) for x in carry)
        cur.append(s)
        cur_tok += t
    if cur:
        tail = " ".join(cur)
        # merge a too-short tail into the previous chunk if that stays within max
        if chunks and _ntok(tail) < min_tokens // 2 and _ntok(chunks[-1]) + _ntok(tail) <= max_tokens * 1.2:
            chunks[-1] = chunks[-1] + " " + tail
        else:
            chunks.append(tail)
    return chunks


def results_to_chunks(results: list[SearchResult], fetch: bool, min_tokens: int,
                      max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    out: list[Chunk] = []
    for r in results:
        text, date = r.content, r.published
        if not text and fetch:
            text, fetched_date = fetch_text(r.url)
            date = date or fetched_date
        if not text:
            text = r.snippet
        if not text:
            continue
        for i, piece in enumerate(chunk_text(text, min_tokens, max_tokens, overlap_tokens)):
            cid = hashlib.sha1(f"{r.url}#{i}".encode()).hexdigest()[:12]
            out.append(Chunk(text=piece, url=r.url, title=r.title, published=date, chunk_id=cid))
    return out
