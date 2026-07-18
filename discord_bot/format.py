"""Pure formatting helpers for Discord output — no discord.py import, so
they're unit-testable. bot.py turns these into an actual discord.Embed.
"""

from __future__ import annotations

from .client import Hit, SearchResult

# Discord limits: embed description ≤ 4096, field value ≤ 1024, total ≤ 6000.
# Keep answers readable and well under the cap.
ANSWER_MAX = 1400
MAX_HIT_LINES = 4


def strip_markdown_bold(text: str) -> str:
    """The API answers use **bold**; Discord uses the same syntax, so leave
    it — but collapse stray triple+ asterisks that would render oddly."""
    return text.replace("***", "**")


def truncate_answer(answer: str, limit: int = ANSWER_MAX) -> str:
    answer = strip_markdown_bold(answer.strip())
    if len(answer) <= limit:
        return answer
    cut = answer[:limit]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 0 else cut).rstrip() + "…"


def format_hits(hits: list[Hit], limit: int = MAX_HIT_LINES) -> str:
    """Markdown list of clickable 'timestamp — title' lines. Discord renders
    [label](url) as a link and auto-suppresses the embed for these."""
    lines = []
    seen: set[str] = set()
    for h in hits:
        if not h.deep_link or h.deep_link in seen:
            continue
        seen.add(h.deep_link)
        ts = h.timestamp or "▶"
        title = _shorten(h.title, 60)
        lines.append(f"[`{ts}` · {title}]({h.deep_link})")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _shorten(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def build_answer_payload(query: str, result: SearchResult) -> dict:
    """A dict the bot converts to an embed. Keeps the (testable) content
    decisions here and the discord.py wiring in bot.py."""
    if result.refused or not result.answer:
        return {
            "title": None,
            "description": ("I couldn't find that in the episodes I've indexed. "
                            "Try rephrasing, or ask about something discussed "
                            "on the show."),
            "hits": "",
            "empty": True,
        }
    return {
        "title": _shorten(query, 240),
        "description": truncate_answer(result.answer),
        "hits": format_hits(result.hits),
        "empty": False,
    }
