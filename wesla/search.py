"""On-demand web search (proposal §4.B).

Providers are tried in order with exponential backoff. `ddgs` needs no key;
Tavily needs TAVILY_API_KEY and is best kept for verification queries.
`StaticSearch` returns canned results (tests / offline demos).
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Protocol

from .log import get_logger
from .types import SearchResult

log = get_logger("search")


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


def _parse_date(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


class DDGSSearch:
    name = "ddgs"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            from ddgs import DDGS  # current package name
        except ImportError:
            try:  # older name
                from duckduckgo_search import DDGS  # type: ignore
            except ImportError:
                raise RuntimeError("ddgs is not installed - run: pip install ddgs") from None
        with DDGS() as d:
            rows = list(d.text(query, max_results=max_results))
        return [SearchResult(title=r.get("title", ""), url=r.get("href") or r.get("url", ""),
                             snippet=r.get("body", "")) for r in rows if r.get("href") or r.get("url")]


class TavilySearch:
    name = "tavily"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY not set")
        import requests

        r = requests.post("https://api.tavily.com/search", timeout=30, json={
            "api_key": self.api_key, "query": query, "max_results": max_results,
            "include_raw_content": True,
        })
        r.raise_for_status()
        return [SearchResult(title=x.get("title", ""), url=x["url"], snippet=x.get("content", ""),
                             content=x.get("raw_content") or "",
                             published=_parse_date(x.get("published_date")))
                for x in r.json().get("results", [])]


class StaticSearch:
    """Returns fixed results for any query. Used by tests and the offline demo."""
    name = "static"

    def __init__(self, results: list[SearchResult] | dict[str, list[SearchResult]]):
        self.results = results

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        if isinstance(self.results, dict):
            for key, res in self.results.items():
                if key.lower() in query.lower():
                    return res[:max_results]
            return []
        return self.results[:max_results]


class SearchChain:
    def __init__(self, providers: list[SearchProvider], max_retries: int = 3, backoff_s: float = 2.0):
        self.providers = providers
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.last_errors: list[str] = []

    def search(self, query: str, max_results: int = 6) -> list[SearchResult]:
        self.last_errors = []
        for p in self.providers:
            for attempt in range(self.max_retries):
                try:
                    t0 = time.perf_counter()
                    res = p.search(query, max_results)
                    log.info("search %s attempt %d: %d result(s) in %.2fs", p.name, attempt + 1, len(res),
                             time.perf_counter() - t0)
                    if res:
                        return res
                    break  # empty result: try next provider
                except Exception as exc:
                    wait = self.backoff_s * (2 ** attempt)
                    log.warning("search %s attempt %d failed: %s (retrying in %.0fs)", p.name, attempt + 1,
                                exc, wait if attempt + 1 < self.max_retries else 0)
                    self.last_errors.append(f"{p.name}#{attempt + 1}: {exc!s:.120}")
                    if attempt + 1 < self.max_retries:
                        time.sleep(wait)
        return []


def make_search(names: list[str], max_retries: int = 3, backoff_s: float = 2.0) -> SearchChain:
    registry = {"ddgs": DDGSSearch, "tavily": TavilySearch}
    providers = [registry[n]() for n in names if n in registry]
    return SearchChain(providers, max_retries, backoff_s)
