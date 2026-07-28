"""Formatting for the concierge bot's Discord output — no discord.py import,
so it stays unit-testable. concierge_bot.py turns these into an Embed.

Reuses format.py's defanging and truncation deliberately: the safety posture
should be identical across both bots, not reimplemented per bot.
"""

from __future__ import annotations

import re

from .concierge_client import ChatResult
from .format import _shorten, defang_urls, truncate_answer

MAX_SOURCES = 5

# Discord allows 4096 in an embed description. format.py's default of 1400 came
# from podcast search, where an answer is a paragraph; a support answer runs to
# about 2000 and was losing its last third to the ellipsis. 3500 leaves room
# for the truncation marker and a little slack.
CONCIERGE_ANSWER_MAX = 3500

_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")

# The concierge answers from documentation and cites page titles, never URLs.
# Unlike podcast search, which exists to link to a YouTube moment, there is no
# case where this bot should post a clickable link — so there is no allowlist
# to get wrong. Every URL in an answer is stripped by truncate_answer.


def flatten_tables(text: str) -> str:
    """Rewrite Markdown tables as plain lines.

    Discord renders no tables at all, so a table arrives as a wall of pipes
    and dashes. The model reaches for one whenever it compares products, which
    is exactly the kind of answer worth reading, so the content is worth
    keeping even though the layout can't survive.

    Each row becomes `first cell — rest`, which is what the table was saying.
    """
    lines = text.split("\n")
    out: list[str] = []
    header: list[str] | None = None

    for line in lines:
        if _TABLE_SEP.match(line):
            continue                       # the |---|---| divider
        row = _TABLE_ROW.match(line)
        if not row:
            header = None
            out.append(line)
            continue
        cells = [c.strip() for c in row.group(1).split("|")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        if header is None:
            header = cells                 # first row of a table is its header
            continue
        label, *rest = cells
        out.append(f"• **{label}** — {' · '.join(rest)}" if rest else f"• **{label}**")

    return "\n".join(out)


def format_sources(titles: list[str], limit: int = MAX_SOURCES) -> str:
    """Documentation pages the answer came from, as plain inline code.

    Titles come from the backend, so they are defanged too: a page named after
    a domain must not render as a link in Discord.
    """
    out: list[str] = []
    for title in titles:
        clean = defang_urls(title).strip()
        if not clean:
            continue
        out.append(f"`{_shorten(clean, 40)}`")
        if len(out) >= limit:
            break
    return " ".join(out)


def build_chat_payload(question: str, result: ChatResult) -> dict:
    """A dict the bot converts to an embed. Content decisions live here; the
    discord.py wiring lives in concierge_bot.py."""
    if result.refused or not result.answer.strip():
        return {
            "title": None,
            "description": (
                "I don't have that in the documentation I've been given. "
                "Try rephrasing, or ask the team directly — I'd rather say "
                "I don't know than guess about your funds."
            ),
            "sources": "",
            "empty": True,
        }
    return {
        # Defang the echoed question too: a user must not be able to make the
        # bot display a URL by putting one in their question.
        "title": _shorten(defang_urls(question), 240),
        "description": truncate_answer(
            flatten_tables(result.answer), CONCIERGE_ANSWER_MAX
        ),
        "sources": format_sources(result.sources),
        "empty": False,
    }
