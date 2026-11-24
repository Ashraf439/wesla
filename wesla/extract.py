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


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def fetch_text(url: str, timeout: float = 6.0) -> tuple[str, datetime | None]:
    """Return (main_text, published_date). Empty string on failure.

    Uses a plain HTTP GET with a short timeout (trafilatura's own downloader waits up to 30 s),
    then trafilatura for boilerplate removal; Jina Reader is the fallback."""
    t0 = time.perf_counter()
    try:
        import requests
        import trafilatura

        r = requests.get(url, timeout=(3.0, timeout), headers=_UA)
        html = r.text if r.ok else ""
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
                log.debug("fetched %s: %d chars in %.2fs", url, len(text), time.perf_counter() - t0)
                return text, date
        log.debug("no extractable text from %s (HTTP %s, %.2fs)", url, r.status_code, time.perf_counter() - t0)
    except Exception as exc:
        log.info("direct fetch failed for %s: %s", url, exc)
    try:
        import requests

        r = requests.get(f"https://r.jina.ai/{url}", timeout=(3.0, timeout),
                         headers={"X-Return-Format": "text"})
        if r.ok and len(r.text) > 200:
            log.debug("fetched %s via Jina Reader: %d chars", url, len(r.text))
            return r.text, None
        log.debug("Jina Reader HTTP %s for %s", r.status_code, url)
    except Exception as exc:
        log.info("Jina Reader failed for %s: %s", url, exc)
    log.info("could not fetch %s after %.2fs; using search snippet", url, time.perf_counter() - t0)
    return "", None


def fetch_all(urls: list[str], budget_s: float = 7.0, timeout: float = 6.0) -> dict[str, tuple[str, datetime | None]]:
    """Fetch pages in parallel. Pages not done within `budget_s` are skipped (their snippet is used)."""
    from concurrent.futures import ThreadPoolExecutor, wait

    if not urls:
        return {}
    pool = ThreadPoolExecutor(max_workers=min(8, len(urls)))
    futs = {pool.submit(fetch_text, u, timeout): u for u in urls}
    done, pending = wait(futs, timeout=budget_s)
    pool.shutdown(wait=False, cancel_futures=True)
    if pending:
        log.info("fetch budget %.0fs reached; %d page(s) skipped: %s", budget_s, len(pending),
                 [futs[f] for f in pending])
    out = {}
    for f in done:
        try:
            out[futs[f]] = f.result()
        except Exception:
            pass
    return out


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
                      max_tokens: int, overlap_tokens: int, budget_s: float = 7.0,
                      timeout: float = 6.0) -> list[Chunk]:
    to_fetch = [r.url for r in results if not r.content] if fetch else []
    fetched = fetch_all(to_fetch, budget_s, timeout)
    out: list[Chunk] = []
    for r in results:
        text, date = r.content, r.published
        if not text and r.url in fetched:
            text, fetched_date = fetched[r.url]
            date = date or fetched_date
        if not text:
            text = r.snippet
        if not text:
            continue
        for i, piece in enumerate(chunk_text(text, min_tokens, max_tokens, overlap_tokens)):
            cid = hashlib.sha1(f"{r.url}#{i}".encode()).hexdigest()[:12]
            out.append(Chunk(text=piece, url=r.url, title=r.title, published=date, chunk_id=cid))
    return out


_W = re.compile(r"[a-z0-9]+")
_STOPW = set("""a an the is are was were be of in on at to for from by with and or what which who how
why when where does do did can could would should this that it its as about into than then""".split())


def select_chunks(chunks: list[Chunk], query: str, max_total: int = 12, max_per_page: int = 3) -> list[Chunk]:
    """Keep the chunks most likely to matter for the query (cheap lexical overlap), at most
    `max_per_page` per page, so verification and the Learning Judge run on a bounded set."""
    if len(chunks) <= max_total:
        return chunks
    q = set(_W.findall(query.lower())) - _STOPW

    def score(c: Chunk) -> float:
        words = _W.findall(c.text.lower())
        if not words or not q:
            return 0.0
        hits = sum(1 for w in words if w in q)
        cover = len(q & set(words)) / len(q)
        return cover + hits / math.sqrt(len(words))

    ranked = sorted(range(len(chunks)), key=lambda i: -score(chunks[i]))
    per: dict[str, int] = {}
    keep: list[int] = []
    for i in ranked:
        u = chunks[i].url
        if per.get(u, 0) >= max_per_page:
            continue
        per[u] = per.get(u, 0) + 1
        keep.append(i)
        if len(keep) >= max_total:
            break
    return [chunks[i] for i in sorted(keep)]
