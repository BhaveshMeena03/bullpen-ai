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
from app.x_api import _URL_SHAPED, Mention, XClient, strip_urls

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "x_bot_state.json"

# X counts the leading @handle of a reply toward the limit in some clients,
# so aim well under 280 rather than discovering the edge in production.
REPLY_BUDGET = 258

# X wraps every URL in t.co and counts it as exactly 23 characters, however
# long it really is (docs.x.com/resources/fundamentals/counting-characters).
# Budgeting the literal length instead threw away thirty characters of answer
# on every reply that carried a link.
URL_WEIGHT = 23
# X API v2 enforces 280 on POST /2/tweets even for Premium accounts, which
# is why replies are written to fit rather than truncated. Configurable in
# case that changes: long-form exists in the product, just not on this
# endpoint today.
POST_LIMIT = 280

# How many times to retry one mention before stepping over it. Three, because
# the failures worth retrying are transient — a timeout, a rate limit, a
# provider blip — and anything that fails three times is a bug that will not
# fix itself before the next poll.
MAX_ATTEMPTS = 3

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


def pinned_answer(question: str, contract_address: str | None,
                  token_label: str | None = None) -> str | None:
    """A fixed reply for questions retrieval should not be asked.

    Returns None when nothing is pinned, so the normal path runs.

    The reply names what the address is for. A bare "CA: 8VjF..." is read
    out of context — quoted, screenshotted, seen weeks later in a reply
    thread — and a 44-character string with nothing attached to it is
    indistinguishable from any other 44-character string someone might post
    under a token account.

    The address is only ever the configured one. It is never read out of the
    incoming post — a bot that echoed back whatever address someone sent it
    would be a ready-made tool for making a scam look endorsed by this
    account.
    """
    if not contract_address or not _ASKS_FOR_CA.search(question or ""):
        return None
    label = token_label or "This project"
    return (f"{label} CA: {contract_address}\n\n"
            f"That is the only official one.")


# People tag an account to say "very cool concept!" far more often than to
# ask it anything. Those are not questions, and answering them is the worst
# case on every axis: the model has nothing to answer so it produces a canned
# deflection, the formatter staples an unrelated citation to it, and with
# links enabled the whole thing costs $0.209 to say nothing.
# Two shapes. Wh-words and asking verbs count anywhere in the text; bare
# auxiliaries only count at the start, where they actually invert a question
# ("is bitcoin mentioned"). Matching "is" anywhere made "this is sick" a
# question.
_ASKING = re.compile(
    r"""(?ix)\b(?: wh(?:at|o|en|ere|y|ich)(?:'?s)? | how(?:'?s)?
                 | tell\s+me | explain | thoughts\s+on
                 | timestamp | quote | search\s+for
    )\b""")
_OPENS_A_QUESTION = re.compile(
    r"""(?ix)^\W*(?: did | does | do | is | are | was | were | has | have
                   | can | could | would | should | any | find | show | give
    )\b""")


# How the bot wants answers, as against how the web page wants them. Sent in
# the user turn rather than the system prompt, so SYSTEM_PROMPT's bytes stay
# identical for every surface and its cache entry keeps working.
#
# Each line here is a reply that actually went wrong. "Your question is
# pretty broad! Could you be more specific?" is a fine thing for a search
# page to say and a wasted $0.209 in a reply thread nobody returns to.
# "I couldn't find a comprehensive summary of everything PoorGoat said, but
# here are the main things" spent a third of the 280 characters before
# reaching the answer.
REPLY_STYLE = """\
This answer will be posted as a single social media reply, not shown on a \
web page. So:
- Under 200 characters. This is a hard limit, not a target: anything longer \
is cut off mid-word, so a complete short answer beats a truncated full one. \
Count as you write.
- One or two sentences. Say the single most concrete thing — a number, a \
name, what somebody actually did — and stop.
- Never ask a follow-up question and never ask the person to be more \
specific. If the question is broad, pick the most striking thing in the \
excerpts and answer with that.
- Do not open by saying what you could not find, and do not open by \
restating the question. Lead with the answer.
- Give the timestamp. The episode name is added for you, so do not repeat \
it."""


def looks_like_a_question(text: str) -> bool:
    """Is this actually asking something?

    A question mark, or an interrogative word. Deliberately generous — a
    false negative costs one unanswered compliment, a false positive costs
    a nonsense reply in public with a link attached.
    """
    text = (text or "").strip()
    if not text:
        return False
    return ("?" in text
            or bool(_ASKING.search(text))
            or bool(_OPENS_A_QUESTION.match(text)))


def is_a_miss(answer: str) -> bool:
    """Did the model say it could not find this?

    The hit list cannot tell you. Retrieval always returns its top_k, so a
    question with no answer in the archive still comes back with six
    passages about something else — which is how the first real reply came
    out as "I couldn't find that" followed by a confident timestamp from an
    unrelated episode.
    """
    return NOT_FOUND_ANSWER.lower() in (answer or "").lower()


def weighted_length(text: str) -> int:
    """Length as X counts it: every URL is 23 characters."""
    total, urls = len(text), 0
    for word in text.split():
        if _URL_SHAPED.search(word):
            total -= len(word)
            urls += 1
    return total + urls * URL_WEIGHT


def format_reply(answer: str, hits: list, include_links: bool = False) -> str:
    """One reply: the answer, then where it was said.

    Two shapes, because what X renders differs.

    With a link, X shows a card carrying the episode title and thumbnail, so
    repeating the title in the text spends fifty characters on something the
    reader can already see. What the card does not show is the moment, which
    is the entire point of this tool — so the text names it, and says
    honestly what the link will do: YouTube lands on the second, X ignores
    timestamps and leaves the viewer to scrub.

    Without a link there is no card, so the title has to be in the text.
    Then the timestamp goes in the tail only when the answer has not already
    given one, because printing both once produced "Around 1:00:00 in the
    episode…" above "1:39:15 ·" — a contradiction in the one detail this
    tool claims to get right.
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
    if include_links:
        seekable = "t=" in (top.deep_link or "")
        # Fitted first, because whether the tail should carry a timestamp
        # depends on whether the trimmed answer already has one — and the
        # tail's own length depends on that answer. Two passes, cheaply.
        def build(cites: bool) -> tuple[str, int]:
            if cites:
                lead = "Watch from there:" if seekable else "Full episode:"
            else:
                lead = (f"Watch from {top.timestamp}:" if seekable
                        else f"Full episode ({top.timestamp}):")
            return lead, POST_LIMIT - len(lead) - 1 - URL_WEIGHT - 2

        lead, budget = build(cites=True)
        fitted = _fit(answer, budget)
        if not _CITES_A_TIME.search(fitted):
            # The answer gave no moment, so the tail has to. Otherwise the
            # reply names a link and no time, which is the one thing this
            # tool is for.
            lead, budget = build(cites=False)
            fitted = _fit(answer, budget)
        return fitted + f"\n\n{lead} {top.deep_link}"

    title = _fit(str(top.title), 60)
    # Assume the answer cites a moment, then check the TRIMMED text rather
    # than the original: deciding on the full answer and trimming afterwards
    # can cut the very timestamp that justified leaving it out.
    tail = f"\n\n{title}"
    fitted = _fit(answer, REPLY_BUDGET - len(tail))
    if not _CITES_A_TIME.search(fitted):
        tail = f"\n\n{top.timestamp} · {title}"
        fitted = _fit(answer, REPLY_BUDGET - len(tail))
    return fitted + tail

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
    # How many times each mention has failed, so a permanently broken one
    # is eventually stepped over instead of blocking everything behind it.
    attempts: dict = field(default_factory=dict)
    spent_usd: float = 0.0
    # Spend is tracked per UTC day as well as cumulatively, because the
    # reply cap does not bound it. Replies are the expensive part but not
    # the only part: every mention read costs $0.001 whether or not it is
    # answered, so anyone willing to tag the account repeatedly can run up a
    # bill without a single reply being sent.
    spent_today_usd: float = 0.0

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
            "attempts": self.attempts,
            "spent_usd": round(self.spent_usd, 4),
            "spent_today_usd": round(self.spent_today_usd, 4),
        }))
        tmp.replace(path)

    def roll(self, today: str) -> None:
        if today != self.day:
            self.day = today
            self.replies_today = 0
            self.spent_today_usd = 0.0


class MentionBot:
    """One poll cycle, with the caps that keep a bug from becoming a bill."""

    def __init__(self, client: XClient, index, *, daily_reply_cap: int = 100,
                 include_links: bool = False, min_question_chars: int = 6,
                 contract_address: str | None = None,
                 daily_spend_cap_usd: float = 5.0,
                 verified_only: bool = False,
                 token_label: str | None = None,
                 state_path: Path = STATE_PATH) -> None:
        self._client = client
        self._index = index
        self.cap = daily_reply_cap
        self.include_links = include_links
        self._min_question = min_question_chars
        self._contract_address = contract_address
        self._spend_cap = daily_spend_cap_usd
        self._verified_only = verified_only
        self._token_label = token_label
        self._state_path = state_path
        self.state = BotState.load(state_path)

    async def tick(self, today: str) -> int:
        """Answer whatever is new. Returns how many replies were posted.

        Every exit records what the cycle cost, so the daily ceiling holds
        on the paths that spend and then return early as well.
        """
        before = self._client.spent_usd
        try:
            return await self._tick(today)
        finally:
            self.state.spent_today_usd += self._client.spent_usd - before
            self.state.spent_usd = round(self._client.spent_usd, 4)
            self.state.save(self._state_path)

    async def _tick(self, today: str) -> int:
        self.state.roll(today)
        if self.state.replies_today >= self.cap:
            logger.info("daily reply cap reached (%d) — idling", self.cap)
            return 0
        if self._spend_cap and self.state.spent_today_usd >= self._spend_cap:
            # The backstop the reply cap is not. Checked before the read,
            # because the read is itself billable and is the part an
            # outsider controls: they choose how often to tag the account.
            logger.warning("daily X spend cap reached ($%.2f) — idling",
                           self._spend_cap)
            return 0

        mentions = await self._client.mentions(since_id=self.state.since_id)
        if not mentions:
            return 0

        if self.state.since_id is None:
            # Cold start with no memory: record where we are and answer
            # nothing this round. Skipping a few questions is recoverable;
            # replying to a backlog all at once is not.
            self.state.since_id = mentions[-1].id
            logger.info("cold start — skipping %d existing mention(s)",
                        len(mentions))
            return 0

        posted = 0
        replied = set(self.state.replied)
        # Advanced only past mentions that were actually dealt with. Setting
        # it per-mention up front meant a failure mid-reply still marked the
        # question as seen, and it was never looked at again: a crash lost a
        # real question silently, which is the one outcome this bot cannot
        # have.
        handled = self.state.since_id
        for mention in mentions:
            if mention.id in replied:
                handled = mention.id
                continue
            if mention.author_id == self._client.bot_user_id:
                handled = mention.id           # never answer itself
                continue
            if self._verified_only and not mention.author_verified:
                # Checked here rather than inside compose(), so an ignored
                # account costs nothing beyond the read that already
                # happened — no retrieval, no model call, no reply.
                logger.info("%s is from an unverified account — skipping",
                            mention.id)
                handled = mention.id
                continue
            if self.state.replies_today + posted >= self.cap:
                logger.info("hit the daily cap mid-batch — stopping")
                break
            try:
                if await self._answer(mention):
                    posted += 1
                    replied.add(mention.id)
                    self.state.replied.append(mention.id)
                handled = mention.id
                self.state.attempts.pop(mention.id, None)
            except Exception:                              # noqa: BLE001
                # Left unhandled so the next poll retries it — but only a
                # few times. A mention that fails every time would otherwise
                # block every question behind it forever.
                tries = self.state.attempts.get(mention.id, 0) + 1
                self.state.attempts[mention.id] = tries
                logger.exception("%s failed (attempt %d/%d)",
                                 mention.id, tries, MAX_ATTEMPTS)
                if tries >= MAX_ATTEMPTS:
                    logger.error("%s failed %d times — giving up on it",
                                 mention.id, tries)
                    handled = mention.id
                    self.state.attempts.pop(mention.id, None)
                break

        self.state.since_id = handled
        self.state.replies_today += posted
        return posted

    async def compose(self, mention: Mention) -> str | None:
        """The reply this mention would get, or None to stay quiet.

        Separate from posting so a reply can be read before it is sent.
        Every reply defect found so far came from looking at composed output
        on a real question rather than from a test.
        """
        question = question_from(mention.text)
        if len(question) < self._min_question:
            logger.info("%s is a tag with no question — skipping", mention.id)
            return None
        # Pinned answers come first, ahead of the question gate: "ca pls" is
        # a request even though it is not shaped like a question, and the
        # contract address is a fact about the project rather than something
        # said on the podcast. Free, instant, and it cannot come back
        # paraphrased.
        pinned = pinned_answer(question, self._contract_address,
                               self._token_label)
        if pinned:
            return pinned

        if not looks_like_a_question(question):
            # Checked before retrieval so a compliment costs nothing at all.
            logger.info("%s is not a question (%r) — staying quiet",
                        mention.id, question[:60])
            return None

        result = await self._index.search(question, instruction=REPLY_STYLE)
        if getattr(result, "refused", False):
            logger.info("%s refused by the model — staying quiet", mention.id)
            return None
        if not is_a_miss(result.answer) and not _CITES_A_TIME.search(result.answer):
            # A real answer from this index always names a moment — the
            # prompt requires it, and citing is the entire point. An answer
            # with no timestamp that is not the honest "couldn't find it" is
            # the model talking about itself ("I appreciate your enthusiasm,
            # but I'm here to answer questions about..."), which reached a
            # reply once with an unrelated episode stapled underneath.
            logger.info("%s produced no citation (%r) — staying quiet",
                        mention.id, result.answer[:70])
            return None
        return format_reply(result.answer, result.hits,
                            include_links=self.include_links) or None

    async def _answer(self, mention: Mention) -> bool:
        text = await self.compose(mention)
        if not text:
            return False
        posted = await self._client.reply(text, mention.id,
                                          allow_link=self.include_links)
        logger.info("replied to %s -> %s", mention.id, posted or "dry run")
        return True

    @staticmethod
    def pause_seconds(base: float = 60.0) -> float:
        """Jitter between polls.

        Perfectly regular intervals are a documented suspension trigger, and
        a bot that answers in the same number of seconds every time reads as
        a bot even when it is welcome.
        """
        return base * random.uniform(0.7, 1.4)
