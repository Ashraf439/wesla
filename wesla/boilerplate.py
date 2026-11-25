"""Detect web-page boilerplate that should never become knowledge.

Navigation menus, cookie/consent banners, reference lists and paper headers survive text
extraction and look "on topic" to an embedder. Stored as trusted memory, they then make the NLI
contradiction check block real content. `boilerplate_reason` returns why a chunk is junk, or None.
"""
from __future__ import annotations

import re

_PHRASES = re.compile(
    r"skip to (main )?content|sign in|log ?in|subscribe|newsletter|cookie|consent|privacy policy|"
    r"terms of (use|service)|all rights reserved|accept all|manage preferences|table of contents|"
    r"follow us|share this|related articles|advertisement|back to top|main menu|toggle navigation|"
    r"demo request|customer support|download the app|javascript|enable (essential )?site functionality",
    re.I)
_CITATION = re.compile(
    r"\bet al\.?|\bpp\. ?\d|\bproceedings\b|\bin: |google scholar|\bdoi\b|arxiv:\s?\d|"
    r"\bvol\. ?\d|\bno\. ?\d|\(\d{4}[a-z]?\)|\bcrossref\b|\bpubmed\b|\bisbn\b", re.I)
_PAPER_HEADER = re.compile(r"\[submitted on|cite as:|subjects:\s|comments:\s|computer science >", re.I)
_SENTENCE = re.compile(r"[^.!?]{25,}[.!?](\s|$)")
_NOT_END = re.compile(r"\bet al\.|\b(?:e\.g|i\.e|vs|approx|fig|eq|sec)\.|\d\.\d", re.I)
# Springer/PubMed-style author lists ("Zhang Q, Chen M, Bukharin A, ..."): each entry ends with a
# period, so they look like prose to _SENTENCE. Real reference lists score 15-25 per 100 words,
# prose stays below 7 on the stored data.
_AUTHOR = re.compile(r"\b[A-Z][a-z]+ [A-Z]{1,2}(?: [A-Z])?[,.]")


def boilerplate_reason(text: str) -> str | None:
    words = text.split()
    n = len(words)
    if n < 12:
        return None  # too short to judge (search snippets); verification handles them
    per100 = 100.0 / n
    phrases = len(_PHRASES.findall(text)) * per100
    cites = len(_CITATION.findall(text)) * per100
    headers = len(_PAPER_HEADER.findall(text))
    # share of the text that sits inside real sentences (menus and lists have almost none)
    # "et al.", "e.g." and decimals ("74.34%") are not sentence ends; without this, academic prose
    # with inline citations looks like a list
    flat = _NOT_END.sub(lambda m: m.group(0).replace(".", ""), text)
    prose = sum(len(m.group(0).split()) for m in _SENTENCE.finditer(flat)) / n
    authors = len(_AUTHOR.findall(text)) * per100
    if headers >= 2:
        return "paper/abstract page header"
    if authors >= 10:
        return f"reference list ({authors:.0f} author names per 100 words)"
    if cites >= 8 or (cites >= 4 and prose < 0.6):
        return f"reference list ({cites:.1f} citation marks per 100 words)"
    if phrases >= 3 or (phrases >= 1.5 and prose < 0.5):
        return f"navigation/consent boilerplate ({phrases:.1f} UI phrases per 100 words)"
    if prose < 0.25 and n >= 40:
        return f"not prose ({prose:.0%} of words in sentences)"
    return None
