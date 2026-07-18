"""Pure formatting helpers for Discord output — no discord.py import, so
they're unit-testable. bot.py turns these into an actual discord.Embed.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .client import Hit, SearchResult

# Discord limits: embed description ≤ 4096, field value ≤ 1024, total ≤ 6000.
# Keep answers readable and well under the cap.
ANSWER_MAX = 1400
MAX_HIT_LINES = 4

# The ONLY domains this bot is ever allowed to link to. Everything the tool
# does is "jump to a moment on YouTube", so any other domain in a result is
# either a bug or a compromise — and must never be posted. This is what makes
# it structurally impossible to turn the bot into a link-drainer via the
# backend/model.
ALLOWED_LINK_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
}

# Anything that Discord would auto-linkify in answer TEXT. The answer is a
# grounded podcast summary — it should never contain a URL, so we strip them.
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[\w-]+\.(?:com|xyz|io|fi|app|net|org|co|gg|link|to|me)\b\S*",
    re.IGNORECASE,
)


def is_allowed_link(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in ALLOWED_LINK_HOSTS


def defang_urls(text: str) -> str:
    """Remove anything that could render as a clickable link in the answer
    body — defense in depth so the answer field can never carry a scam URL,
    even if the backend/model were manipulated into producing one."""
    return _URL_RE.sub("[link removed]", text)


def strip_markdown_bold(text: str) -> str:
    """The API answers use **bold**; Discord uses the same syntax, so leave
    it — but collapse stray triple+ asterisks that would render oddly."""
    return text.replace("***", "**")


def truncate_answer(answer: str, limit: int = ANSWER_MAX) -> str:
    answer = defang_urls(strip_markdown_bold(answer.strip()))
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
        # Only ever post links to the YouTube domains. A result pointing
        # anywhere else is dropped — the bot can't be made to post a
        # non-YouTube link no matter what the backend returns.
        if not h.deep_link or h.deep_link in seen or not is_allowed_link(h.deep_link):
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
        # Defang the echoed question too: a user must not be able to make the
        # bot display a URL via their search text.
        "title": _shorten(defang_urls(query), 240),
        "description": truncate_answer(result.answer),
        "hits": format_hits(result.hits),
        "empty": False,
    }
