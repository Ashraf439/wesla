"""Dynamic RAG prompt construction (proposal §4.C)."""
from __future__ import annotations

SYSTEM = (
    "You are WESLA, a careful assistant. Answer using the numbered CONTEXT passages when they are "
    "relevant and cite them like [1]. The passages were retrieved just now, from the live web or from "
    "memory, so they can describe events after your training cutoff: trust them over what you were "
    "trained on, and never reply that you cannot know about recent or future events when passages "
    "are given. If the context does not contain the answer, say so briefly and answer from general "
    "knowledge, marking it as such. Prefer newer passages when they disagree."
)


def build_messages(query: str, contexts: list[tuple[str, str]], history: str = "",
                   max_ctx_chars: int = 6000) -> list[dict]:
    """contexts: list of (text, source_label)."""
    lines, used = [], 0
    for i, (text, src) in enumerate(contexts, 1):
        piece = f"[{i}] ({src}) {text.strip()}"
        if used + len(piece) > max_ctx_chars:
            piece = piece[: max(0, max_ctx_chars - used)]
        if not piece:
            break
        lines.append(piece)
        used += len(piece)
    user = ""
    if history:
        user += f"CONVERSATION SO FAR:\n{history}\n\n"
    user += "CONTEXT:\n" + ("\n".join(lines) if lines else "(none)") + f"\n\nQUESTION: {query}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
