"""Answer questions that tag the search account on X.

Someone tags @MarketBubbleSearch with a question, the bot answers from the
indexed transcripts and cites the moment. That shape is deliberate: X
restricted programmatic replies in February 2026 so that an app may only
reply when the author mentioned or quoted it first. Generic reply bots
stopped working; tag-to-ask is precisely the case that still does.

Three things decide the design.

Replies carry no link. A post containing a URL costs $0.200 against $0.015
without one — thirteen times, and X does not document what their detector
counts, so the safe rule is to carry no URL at all. The quote and the
timestamp are the useful part anyway; the link lives in the bio. Set
`x_bot_include_links` if someone else is paying, which is the only reason
to.

Retrieval happens in-process. Calling the public search endpoint over HTTP
would put the bot behind this service's own rate limiter — 200 requests per
IP per day, twelve a minute — and the bot is a single IP on Render. It would
throttle itself by lunchtime and take real visitors' budget with it.

Every guard is a spend guard. Each poll and each reply costs money, so a bug
that loops is not a wrong answer, it is a bill. Hence the daily cap, the
replied-set, and a cold start that skips the backlog rather than answering
it twice.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.podcast import NOT_FOUND_ANSWER
from app.x_api import Mention, XClient, strip_urls

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "x_bot_state.json"

# X counts the leading @handle of a reply toward the limit in some clients,
# so aim well under 280 rather than discovering the edge in production.
REPLY_BUDGET = 258

_HANDLE = re.compile(r"@\w{1,15}")
# A spoken-timestamp citation in the answer text: 16:16, 1:39:15, 4:01:47.
_CITES_A_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_WHITESPACE = re.compile(r"\s+")


def question_from(text: str) -> str:
    """The question inside a post that tagged the bot.

    Strips every handle, not just the bot's: a post reading "@marketbubble
    @searchbot what did ansem say about eth" is asking about Ansem, and
    leaving the handles in sends them to the embedder as if they were
    search terms.
    """
    return _WHITESPACE.sub(" ", _HANDLE.sub(" ", text)).strip()


def _fit(text: str, budget: int) -> str:
    """Trim to budget on a sentence boundary if there is one, else a word."""
    text = text.strip()
    if len(text) <= budget:
        return text
    cut = text[:budget]
    for boundary in (". ", "! ", "? "):
        at = cut.rfind(boundary)
        if at > budget * 0.55:
            return cut[:at + 1].strip()
    at = cut.rfind(" ")
    return (cut[:at] if at > 0 else cut).rstrip(",;:") + "…"


# Markdown and the excerpt format leak into answers, because the prompt was
# written for a web page that renders both. X renders neither: "**Tokenomics**"
# shows its asterisks, and "[1:39:32]" — the marker each transcript line
# carries so the model can cite the line it used — reads as broken markup.
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_CODE = re.compile(r"`([^`]*)`")
_BRACKET_TIME = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")
_LIST_MARK = re.compile(r"(?m)^\s*[-*+]\s+|^#{1,6}\s+")


def plain_text(answer: str) -> str:
    """Strip web-page formatting a plain-text reply cannot render."""
    text = _BRACKET_TIME.sub(r"\1", answer or "")
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    text = _CODE.sub(r"\1", text)
    text = _LIST_MARK.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


# "what's the CA", "contract address?", "drop the mint". Asked constantly
# under a token account, and retrieval is the wrong tool for it: the address
# is a fact about the project, not something anyone said on the podcast.
# Answering from a constant is instant, costs nothing, and cannot be got
# subtly wrong — which for a contract address is the only acceptable bar.
_ASKS_FOR_CA = re.compile(
    r"""(?ix)
    \b(?: ca                              # bare "ca", not "california"
         | contract(?:\s+address)?
         | token\s+address
         | mint(?:\s+address)?
         | address
    )\b""")


def pinned_answer(question: str, contract_address: str | None) -> str | None:
    """A fixed reply for questions retrieval should not be asked.

    Returns None when nothing is pinned, so the normal path runs.

    The address is only ever the configured one. It is never read out of the
    incoming post — a bot that echoed back whatever address someone sent it
    would be a ready-made tool for making a scam look endorsed by this
    account.
    """
    if not contract_address or not _ASKS_FOR_CA.search(question or ""):
        return None
    return f"CA: {contract_address}"


def is_a_miss(answer: str) -> bool:
    """Did the model say it could not find this?

    The hit list cannot tell you. Retrieval always returns its top_k, so a
    question with no answer in the archive still comes back with six
    passages about something else — which is how the first real reply came
    out as "I couldn't find that" followed by a confident timestamp from an
    unrelated episode.
    """
    return NOT_FOUND_ANSWER.lower() in (answer or "").lower()


def format_reply(answer: str, hits: list, include_links: bool = False) -> str:
    """One reply: the answer, then where it was said.

    The citation is the point. Anyone can paraphrase an episode; naming the
    second it happened is the thing this index can do and a person scrolling
    cannot — which is exactly why it must not be attached to an answer that
    found nothing. A timestamp on "I couldn't find that" is worse than no
    citation: it reads as a real source and points somewhere unrelated.
    """
    answer = plain_text(strip_urls(answer))   # guests read links aloud;
                                             # the model writes markdown
    if is_a_miss(answer):
        # Just the sentence. The model tends to follow it with an offer to
        # try another question, which is fine on a web page and reads as
        # padding in a reply — and gets cut mid-word by the length budget.
        return NOT_FOUND_ANSWER + "."
    if not hits:
        return _fit(answer, REPLY_BUDGET)

    top = hits[0]
    title = _fit(str(top.title), 60)
    # The model cites the line it actually used; hits[0].timestamp is where
    # that passage begins, and the two are often minutes apart. Printing
    # both put "Around 1:00:00" above "1:39:15 ·" in the same reply. When
    # the answer already names a moment, the tail carries only the episode.
    tail = (f"\n\n{title}" if _CITES_A_TIME.search(answer)
            else f"\n\n{top.timestamp} · {title}")
    if include_links:
        # Only when someone else is funding it: this makes every reply cost
        # $0.200 instead of $0.015.
        tail += f"\n{top.deep_link}"
    return _fit(answer, REPLY_BUDGET - len(tail)) + tail


@dataclass
class BotState:
    """What must survive a restart so nobody gets answered twice.

    Render's disk does not persist, so this can come back empty. That is
    handled by treating a missing `since_id` as "start from now" rather than
    "answer everything ever posted" — a bot that replies to a month of old
    mentions at once is the exact pattern that gets accounts suspended.
    """

    since_id: str | None = None
    replied: list[str] = field(default_factory=list)
    day: str = ""
    replies_today: int = 0
    spent_usd: float = 0.0

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> BotState:
        if not path.exists():
            return cls()
        try:
            return cls(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("x_bot state unreadable (%s) — starting fresh", exc)
            return cls()

    def save(self, path: Path = STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "since_id": self.since_id,
            # Bounded: only recent ids matter, since `since_id` already
            # stops anything older from being read again.
            "replied": self.replied[-500:],
            "day": self.day,
            "replies_today": self.replies_today,
            "spent_usd": round(self.spent_usd, 4),
        }))
        tmp.replace(path)

    def roll(self, today: str) -> None:
        if today != self.day:
            self.day = today
            self.replies_today = 0


class MentionBot:
    """One poll cycle, with the caps that keep a bug from becoming a bill."""

    def __init__(self, client: XClient, index, *, daily_reply_cap: int = 100,
                 include_links: bool = False, min_question_chars: int = 6,
                 contract_address: str | None = None,
                 state_path: Path = STATE_PATH) -> None:
        self._client = client
        self._index = index
        self.cap = daily_reply_cap
        self.include_links = include_links
        self._min_question = min_question_chars
        self._contract_address = contract_address
        self._state_path = state_path
        self.state = BotState.load(state_path)

    async def tick(self, today: str) -> int:
        """Answer whatever is new. Returns how many replies were posted."""
        self.state.roll(today)
        if self.state.replies_today >= self.cap:
            logger.info("daily reply cap reached (%d) — idling", self.cap)
            return 0

        mentions = await self._client.mentions(since_id=self.state.since_id)
        if not mentions:
            return 0

        if self.state.since_id is None:
            # Cold start with no memory: record where we are and answer
            # nothing this round. Skipping a few questions is recoverable;
            # replying to a backlog all at once is not.
            self.state.since_id = mentions[-1].id
            self.state.save(self._state_path)
            logger.info("cold start — skipping %d existing mention(s)",
                        len(mentions))
            return 0

        posted = 0
        replied = set(self.state.replied)
        for mention in mentions:
            self.state.since_id = mention.id
            if mention.id in replied:
                continue
            if mention.author_id == self._client.bot_user_id:
                continue                       # never answer itself
            if self.state.replies_today + posted >= self.cap:
                logger.info("hit the daily cap mid-batch — stopping")
                break
            if await self._answer(mention):
                posted += 1
                replied.add(mention.id)
                self.state.replied.append(mention.id)

        self.state.replies_today += posted
        self.state.spent_usd = round(self._client.spent_usd, 4)
        self.state.save(self._state_path)
        return posted

    async def _answer(self, mention: Mention) -> bool:
        question = question_from(mention.text)
        if len(question) < self._min_question:
            logger.info("%s is a tag with no question — skipping", mention.id)
            return False

        # Before retrieval, because the contract address is a fact about the
        # project rather than something said on the podcast. Answering it
        # from a constant costs nothing, cannot be paraphrased wrong, and
        # skips the model entirely.
        pinned = pinned_answer(question, self._contract_address)
        if pinned:
            posted = await self._client.reply(pinned, mention.id)
            logger.info("replied to %s with the pinned CA -> %s",
                        mention.id, posted or "dry run")
            return True

        result = await self._index.search(question)
        if getattr(result, "refused", False):
            logger.info("%s refused by the model — staying quiet", mention.id)
            return False

        text = format_reply(result.answer, result.hits,
                            include_links=self.include_links)
        if not text:
            return False
        posted = await self._client.reply(text, mention.id)
        logger.info("replied to %s (%s hits) -> %s",
                    mention.id, len(result.hits), posted or "dry run")
        return True

    @staticmethod
    def pause_seconds(base: float = 60.0) -> float:
        """Jitter between polls.

        Perfectly regular intervals are a documented suspension trigger, and
        a bot that answers in the same number of seconds every time reads as
        a bot even when it is welcome.
        """
        return base * random.uniform(0.7, 1.4)
