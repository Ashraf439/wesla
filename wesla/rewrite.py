"""Follow-up question rewriting.

"What rank is typically used for it?" means nothing to the router, the memory search or a web
search engine. When a query refers back to the conversation (pronouns, "the same", very short
follow-ups), the LLM rewrites it into a standalone question using the previous user turns.
"""
from __future__ import annotations

import re

from .activity import Stopped
from .log import get_logger

log = get_logger(__name__)

_REF = re.compile(r"\b(it|its|it's|they|them|their|this|that|these|those|he|she|him|her|his|"
                  r"the same|the former|the latter|one|ones|there)\b", re.I)
_PROMPT = ("Rewrite the FOLLOW-UP as one standalone question that names what it refers to, using the "
           "EARLIER QUESTIONS. Keep the meaning. If it is already standalone, return it unchanged. "
           "Output only the question.\n\n"
           "Example 1\nEARLIER QUESTIONS:\n- what is LoRA in fine-tuning?\n"
           "FOLLOW-UP: what rank is typically used for it?\n"
           "what rank is typically used for LoRA in fine-tuning?\n\n"
           "Example 2\nEARLIER QUESTIONS:\n- who founded Nvidia?\n"
           "FOLLOW-UP: when was that?\nwhen was Nvidia founded?\n\n"
           "Example 3\nEARLIER QUESTIONS:\n- what is QLoRA?\n"
           "FOLLOW-UP: does it work on a T4 GPU?\ndoes QLoRA work on a T4 GPU?")
# "this week", "that year" point at time, not back at the conversation
_TIME = re.compile(r"\b(this|that|these|those)\s+(morning|evening|week|weekend|month|year|quarter|season|"
                   r"time|moment)\b", re.I)


_STOP = set("""a an the is are was were be of in on at to for from by with and or what which who how why
when where does do did can could would should will this that these those it its they them their
there one ones same former latter he she him her his about also so""".split())


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def needs_rewrite(query: str, previous: list[str]) -> bool:
    """A follow-up either points back ("it", "that one") or is too short to stand alone
    ("what rank is typically used?", "and for vision models?")."""
    if not previous:
        return False
    n = len(query.split())
    return n <= 20 and (bool(_REF.search(_TIME.sub(" ", query))) or n <= 6)


def _topic_fallback(query: str, previous: list[str]) -> str:
    """When the model cannot rewrite, attach the previous question's topic words, which is
    enough for memory search, web search and the router to find the right subject."""
    have = _content(query)
    words = [w for w in re.findall(r"[A-Za-z0-9][\w\-.+]*", previous[-1])
             if w.lower() not in _STOP and w.lower() not in have and len(w) > 1]
    if not words:
        return query
    out = f"{query.rstrip()} ({' '.join(dict.fromkeys(words))[:80]})"
    log.info("rewrite: fallback %r -> %r", query, out)
    return out


def rewrite_followup(llm, query: str, previous: list[str]) -> str:
    """Return a standalone version of `query`, or `query` itself if rewriting fails or looks wrong."""
    if not needs_rewrite(query, previous):
        return query
    earlier = "\n".join(f"- {q}" for q in previous[-3:])
    try:
        gen = llm.chat([{"role": "system", "content": _PROMPT},
                        {"role": "user", "content": f"EARLIER QUESTIONS:\n{earlier}\n\nFOLLOW-UP: {query}"}],
                       max_tokens=60, temperature=0.0)
    except Stopped:
        raise
    except Exception as exc:
        log.warning("rewrite: failed (%s), using the topic fallback", exc)
        return _topic_fallback(query, previous)
    out = gen.text.strip().strip('"').splitlines()[0].strip() if gen.text.strip() else ""
    # the prompt's own labels ("FOLLOW-UP:", "EARLIER QUESTIONS:") are often echoed back
    out = re.sub(r"^(?:(?:standalone|rewritten|follow[- ]?up|earlier)(?: questions?)?|questions?|rewritten)\s*:\s*",
                 "", out, flags=re.I).strip('" ')
    # guard against the small model answering instead of rewriting, or rambling
    if not out or len(out.split()) > 3 * len(query.split()) + 15:
        log.info("rewrite: rejected %r", out[:120])
        return _topic_fallback(query, previous)
    # a real rewrite only adds the missing topic: it must keep most of the follow-up's own words
    own = _content(query)
    if own and len(own & _content(out)) / len(own) < 0.6:
        log.info("rewrite: rejected %r (drops the question's own words)", out[:120])
        return _topic_fallback(query, previous)
    if out != query:
        log.info("rewrite: %r -> %r", query, out)
    return out
