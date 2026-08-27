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

import hashlib
import json
import logging
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.podcast import NOT_FOUND_ANSWER
from app.x_api import _URL_SHAPED, Mention, XClient, strip_urls

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
def _state_path() -> Path:
    """Where the bot remembers what it has answered.

    Beside the data files when that is writable, which it is locally and
    makes the file easy to inspect. In the container it is not: the image
    ships at /srv with no writable data directory, so every save raised
    PermissionError — and because the save runs in a finally, it took the
    whole poll cycle down with it, every twenty seconds, silently. The bot
    looked alive, healthz was green, and it had stopped answering.

    Falling back to the temp directory rather than failing: losing this file
    is already an expected condition, since the disk is ephemeral and every
    deploy clears it. The cold start is built for exactly that.
    """
    preferred = ROOT / "data"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write-test"
        probe.touch()
        probe.unlink()
        return preferred / "x_bot_state.json"
    except OSError:
        return Path(tempfile.gettempdir()) / "x_bot_state.json"


STATE_PATH = _state_path()

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

# How far back a cold start will still answer. Render's disk is ephemeral, so
# a deploy hands the bot an empty state file and it has to decide what to do
# with everything already waiting. Thirty minutes is comfortably longer than
# a deploy and far shorter than a backlog.
COLD_START_GRACE = 30 * 60


def _is_recent(created_at: str, now: float | None = None) -> bool:
    """Was this posted inside the cold-start grace window?"""
    if not created_at:
        return False                       # unknown age: treat as old
    try:
        when = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    age = (now or time.time()) - when.timestamp()
    return 0 <= age <= COLD_START_GRACE


def _before(mention_id: str) -> str:
    """The id just below this one, so a since_id lands before it."""
    return str(max(0, int(mention_id) - 1))

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
# Single moments and ranges alike. The range form leaked into a live
# reply as "[2:29:34–2:33:04]", because the pattern only knew about one
# timestamp — and the model writes ranges whenever an answer spans a
# stretch of conversation, which a longer reply does constantly.
_STAMP = r"\d{1,2}:\d{2}(?::\d{2})?"
_BRACKET_TIME = re.compile(
    rf"\[({_STAMP})\s*(?:[-–—]\s*({_STAMP}))?\]")
_LIST_MARK = re.compile(r"(?m)^\s*[-*+]\s+|^#{1,6}\s+")


# X's rules: "Don't Direct Message, mention, or reply to users with
# potentially sensitive content (including profanity), unless they've clearly
# indicated an intent to receive it in advance." Someone asking what Ansem
# said about Ethereum has not indicated any such thing.
#
# The transcripts are full of it — 2,689 lines — because it is a live crypto
# show, and an answer quoting one of those lines put the word in a reply to a
# stranger. Masked rather than dropped: the quote stays faithful, and the
# reader can see exactly what was said without this account being the one
# that said it.
_PROFANITY = re.compile(
    r"""(?ix)\b(?: f+u+c+k | sh+i+t | bitch | cunt | dick(?:head)?
                 | asshole | bastard | wank\w* | prick | tw?at
    )(\w*)\b""")


def soften(text: str) -> str:
    """Mask profanity, keeping the first letter and any suffix."""
    def mask(m: re.Match) -> str:
        word = m.group(0)
        tail = m.group(1) or ""
        core = word[:len(word) - len(tail)]
        return word[0] + "*" * (len(core) - 1) + tail
    return _PROFANITY.sub(mask, text or "")


def plain_text(answer: str, keep_breaks: bool = False) -> str:
    """Strip web-page formatting a plain-text reply cannot render.

    Line breaks are collapsed by default, because a two-sentence answer that
    arrives with stray newlines reads as broken. `keep_breaks` is for the
    long form: a three-thousand-character summary flattened into one
    paragraph is a wall nobody reads, and the paragraph breaks are most of
    what makes it legible.
    """
    text = _BRACKET_TIME.sub(
        lambda m: m.group(1) + (f"–{m.group(2)}" if m.group(2) else ""),
        answer or "")
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    text = _CODE.sub(r"\1", text)
    text = _LIST_MARK.sub("", text)
    if not keep_breaks:
        return _WHITESPACE.sub(" ", text).strip()
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    # Collapse runs of blank lines to one, so the spacing is even however
    # the model laid it out.
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


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


# "summarise episode 14", "summary of ep 12", "what happened in #9". The
# episode summaries already exist and already carry timestamps, so this is a
# lookup rather than a question — no retrieval, no model call, and the answer
# cannot come back different from the one on the website.
_ASKS_FOR_SUMMARY = re.compile(
    r"""(?ix)\b(?: summar (?:ise|ize|y) | recap | rundown | what\s+happened )\b
        [^0-9]{0,40}
        (?: ep(?:isode)?\s*)? \#? \s* (\d{1,2}) \b""")


def summary_request(question: str) -> int | None:
    """The episode number someone is asking to have summarised, or None."""
    found = _ASKS_FOR_SUMMARY.search(question or "")
    return int(found.group(1)) if found else None


def episode_number(title: str) -> int | None:
    """The show's own number for an episode, from its title."""
    found = re.search(r"(?ix)(?: market\s+bubble | ep(?:isode)? )\s*\#?\s*(\d{1,2})\b",
                      title or "")
    return int(found.group(1)) if found else None


# "TL;DR —" is the first thing anyone sees in a summary reply, and it spends
# characters telling them what they already know: they asked for a summary.
# Stripped here rather than regenerating thirty-two summaries to remove four
# characters, and the website keeps it, where a labelled block is useful
# scanning down a page.
_TLDR = re.compile(r"(?i)^\s*(?:\*\*)?tl;?\s*dr(?:\*\*)?\s*[—–:-]*\s*")


# A line that opens with a timestamp is a topic entry.
_TOPIC_LINE = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$")


def _space_out(body: str) -> str:
    """Give each timestamped topic its own block.

    Run together, a dozen entries of three lines each are a wall with
    nowhere for the eye to land — every timestamp is buried mid-paragraph
    where it reads as part of the sentence before it. A blank line between
    them turns each timestamp into an anchor you can scan down, which is the
    only way anyone finds the bit they came for.

    The separator after the time does the same job at word level: it stops
    "0:07:15 Discussion of" parsing as one phrase.
    """
    out: list[str] = []
    for line in body.splitlines():
        entry = _TOPIC_LINE.match(line.strip())
        if entry:
            if out and out[-1]:
                out.append("")
            out.append(f"{entry.group(1)} · {entry.group(2)}")
        else:
            out.append(line)
    return "\n".join(out)


def format_summary(summary: str, title: str, limit: int,
                   url: str | None = None) -> str:
    """A stored summary, as a reply.

    The link is worth its $0.200 here in a way it is not on a two-sentence
    answer. A summary is what someone reads when deciding whether to watch
    the episode at all, so the thing to hand them next is the episode — and
    X renders it as a card with the title and thumbnail, which is most of
    what makes a wall of text look like something rather than a dump.

    The title is dropped when a link is present: the card already shows it.
    """
    body = _space_out(
        _TLDR.sub("", plain_text(soften(strip_urls(summary)),
                                 keep_breaks=True)))
    if url:
        tail = f"\n\nFull episode:\n{url}"
        return _fit(body, limit - URL_WEIGHT - 18) + tail
    head = _fit(str(title), 70)
    return _fit(body, limit - len(head) - 2) + f"\n\n{head}"


# X forbids "duplicative or substantially similar posts on one account", and
# the fixed replies are the only ones that repeat: a generated answer differs
# every time, while the contract address was byte-identical however many
# people asked. Twenty identical posts is the shape that rule describes.
#
# Varied by hashing the question rather than at random, so the same person
# asking twice gets the same answer — consistency where it matters — while
# twenty different people get twenty different phrasings.
_CA_PHRASINGS = (
    "{label} CA: {ca}\n\nThat is the only official one.",
    "{label} contract: {ca}\n\nAccept no other.",
    "The CA for {label} is {ca}\n\nAnything else is not us.",
    "{ca}\n\nThat is the {label} contract address, and the only one.",
    "Official {label} CA:\n{ca}\n\nThere is no other.",
)

_MISS_PHRASINGS = (
    NOT_FOUND_ANSWER + ".",
    NOT_FOUND_ANSWER + " — it may be in a part I have not indexed yet.",
    "I looked, and " + NOT_FOUND_ANSWER[2:].lower() + ".",
    NOT_FOUND_ANSWER + ". Try naming the guest or the episode?",
)


def _pick(options: tuple, seed: str) -> str:
    """Choose deterministically from `seed`, so a repeat is consistent."""
    return options[int(hashlib.sha256(seed.encode()).hexdigest(), 16)
                   % len(options)]


# "what is this", "what do you do", "who are you". Obvious in hindsight and
# badly handled: asked what it was, the bot searched the transcripts for an
# answer, found nothing, and said "I couldn't find that in the episodes I've
# indexed" — which is the one reply guaranteed to make it look broken to
# someone deciding whether it works.
_ASKS_WHAT_THIS_IS = re.compile(
    r"""(?ix)
    (?: wh(?:at|o)(?:'?s|\ is|\ are|\ the\ hell\ is|\ tf\ is)?\s+
        (?: this | that | it | you | u | mbubble\w* | marketbubblesearch
          | your\ (?:deal|purpose) )\b
      | what\ (?:do|can)\s+(?:you|u|this|it)\s+do
      | how\s+(?:do(?:es)?\s+)?(?:you|this|it)\s+work
      | explain\s+(?:yourself|this)
      | wtf\ is\ (?:this|that|it)
    )""")


def about_answer(question: str, site: str | None = None) -> str | None:
    """What this account is, when someone asks.

    A fixed answer rather than a retrieved one, for the same reason as the
    contract address: it is a fact about the project, not something anyone
    said on the podcast, and the index has nothing to say about it.
    """
    if not _ASKS_WHAT_THIS_IS.search(question or ""):
        return None
    body = f"{_pick(_ABOUT_PHRASINGS, question)}\n\n{_ABOUT_ORIGIN}"
    return f"{body}\n\n{site}" if site else body


# No @mention of the operator in any of these. X restricts mentioning
# accounts that are not already in the thread, and a reply that tags someone
# uninvolved is the kind of thing the automation rules are written about.
_ABOUT_PHRASINGS = (
    "I'm a semantic search engine over the entire Market Bubble archive.\n\n"
    "Ask in plain English — you don't need the exact words anyone used. "
    "Every episode is transcribed and indexed by meaning, so \"why does "
    "ansem think eth is done\" finds the moment even if nobody said it "
    "that way.\n\n"
    "You get the answer and the exact second it was said. Every episode is "
    "in there, plus the full live broadcasts — about a third of each show "
    "never reaches the YouTube upload, and that part is searchable here and "
    "nowhere else.\n\n"
    "I only answer from what was actually said. If it isn't in the "
    "archive I'll tell you so rather than guess.",

    "Semantic search across every Market Bubble episode.\n\n"
    "Not keyword matching — the transcripts are indexed by meaning, so you "
    "can ask the way you'd ask a person and it finds the moment even when "
    "the words don't line up.\n\n"
    "Ask me anything from any episode and you get the answer plus the "
    "timestamp it was said at. That includes the live broadcasts, which run "
    "about a third longer than the uploads — the Squire founder interview, "
    "for instance, starts 26 minutes after the ep 10 video ends.\n\n"
    "Everything is grounded in the transcripts. No guessing.",

    "I've transcribed and indexed every Market Bubble episode, then made it "
    "searchable by meaning rather than by keyword.\n\n"
    "So you can ask \"what did luca netz say about pudgy penguins\" without "
    "knowing which episode, and get back what he said and the second he "
    "said it.\n\n"
    "The live broadcasts are indexed too, which is the part nobody else "
    "has — roughly a third of every show is cut before it reaches YouTube.\n\n"
    "I answer only from the transcripts, and say so when something isn't "
    "in there.",
)

_ABOUT_ORIGIN = ("Built for the AnsemHack Clawrena, and for the Market "
                 "Bubble and Bullpen ecosystem.")


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
    return _pick(_CA_PHRASINGS, question).format(
        label=label, ca=contract_address)


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
def reply_style(limit: int = POST_LIMIT) -> str:
    """The answering style, sized to whatever the account can actually post.

    Each rule here is a reply that went wrong. "Your question is pretty
    broad! Could you be more specific?" is fine on a search page and a
    wasted $0.209 in a thread nobody returns to. "I couldn't find a
    comprehensive summary of everything PoorGoat said, but here are the main
    things" spent a third of the characters before reaching the answer.
    """
    budget = max(120, int(limit * 0.72))      # leaves room for the tail
    length = (f"- Under {budget} characters. This is a hard limit, not a "
              "target: anything longer is cut off mid-word, so a complete "
              "short answer beats a truncated full one. Count as you write.\n"
              "- One or two sentences. Say the single most concrete thing — "
              "a number, a name, what somebody actually did — and stop."
              if limit <= 400 else
              f"- Under {budget} characters, which is room for real detail. "
              "Use it: quote what was actually said, give the numbers, name "
              "the people. Do not pad to fill it either — stop when the "
              "answer is complete.\n"
              "- Break it into short paragraphs with a blank line between "
              "each. One unbroken block of 800 characters is a wall on a "
              "phone and nobody reads to the end of it.")
    return f"""\
This answer will be posted as a social media reply, not shown on a web page.
So:
{length}
- Never ask a follow-up question and never ask the person to be more \
specific. If the question is broad, pick the most striking thing in the \
excerpts and answer with that.
- Do not open by saying what you could not find, and do not open by \
restating the question. Lead with the answer.
- The transcripts contain a lot of swearing. Paraphrase around it rather \
than quoting it — the person asking has not asked to be sworn at.
- Give the timestamp. The episode name is added for you, so do not repeat \
it."""


# Kept for callers that want the default shape.
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


# X requires "a clear and easy way for users to opt-out of receiving
# automated replies, and promptly honor all such opt-out requests". One word
# in a reply, matched generously — someone asking to be left alone should
# never have to guess the magic phrase.
# Unambiguous phrases: these mean one thing wherever they appear.
_OPT_OUT = re.compile(
    r"""(?ix)\b(?: unsubscribe | opt\s*-?\s*out | leave\s+me\s+alone
                 | don'?t\s+(?:reply|respond|message|tag|@)
                 | stop\s+(?:replying|responding|tagging|messaging)
                 | no\s+more\s+(?:replies|messages|bots?)
                 | mute\s+me | remove\s+me | unfollow\s+me
    )\b""")
# A bare "stop" is only an opt-out when it is the whole message. Matching it
# anywhere turned "what did ansem say about the stop loss" into a permanent
# block on a real person, and a false positive here is much worse than a
# false negative: one silences someone who wanted an answer, the other means
# they have to say it more plainly.
_BARE_STOP = re.compile(r"(?i)^\W*(?:stop|quiet|shush)\W*$")


def asks_to_be_left_alone(text: str) -> bool:
    """Is this person asking not to be replied to again?"""
    text = (text or "").strip()
    if _OPT_OUT.search(text):
        return True
    return bool(_BARE_STOP.match(question_from(text)))


# Compliments, greetings and reactions. Short, enumerable, and the whole
# reason the gate exists.
_PLEASANTRY = re.compile(
    r"""(?ix)^\W*(?:
        g[mn] | hi | hey | yo | lfg | based | dub | fire | goat | gg | ty
      | thanks? | thank\s+you | congrats\w* | welcome | respect | salute
      | (?:this|that|it)\s+is | looks? | seems? | feels?
      | you\s+(?:beauty|legend|genius|star|beaut)
      # "you are so freaking cool", "you're a genius", "that's sick"
      | (?:you'?re | you\s+are | that'?s | thats | these\s+are)\b
      | (?:absolute|actual)\s+\w+ | let'?s\s+go | no\s+way | holy
      | (?:very|so|really|pretty|super|quite)
      | (?:good|nice|great|solid|clean|sick|dope|cool|huge|wild|insane)
      | love\s+(?:it|this) | (?:i\s+)?appreciate
      | test(?:ing|ed)?
    )\b""")

# A pleasantry is short. "nice work" is a compliment; "nice breakdown of what
# ansem said about the fee situation" is someone asking about the fee
# situation, and the opening word should not decide that.
_PLEASANTRY_MAX_WORDS = 6


def looks_like_a_question(text: str) -> bool:
    """Should this mention get an answer?

    Written the other way round from how it started. Allow-listing question
    shapes left real queries out — "luca netz pudgy penguins" is a perfectly
    natural way to use this and got silence, and "summarize episode 14" is
    an imperative with no question word in it at all. There are far more
    ways to ask something than to pay a compliment, so the compliments are
    the list worth enumerating.

    Getting this wrong in the answering direction is cheap now: an answer
    with no citation is never posted, so a compliment that slips through
    costs one model call and says nothing in public. Getting it wrong in
    the silent direction is what made the account look broken.
    """
    text = (text or "").strip()
    if not text or len(text.split()) < 2:
        return False
    words = text.split()
    if len(words) <= _PLEASANTRY_MAX_WORDS and _PLEASANTRY.match(text):
        return False
    # Nothing but emoji and punctuation.
    return bool(re.search(r"[a-z0-9]{3}", text, re.I))


# Ways the model says "no" without using the sentence it was told to use.
# Phrase-matching one canonical string kept letting a differently-worded
# deflection through: "I don't have enough information to answer this
# question. The excerpts provided don't contain..." went out under a real
# post, with a citation and a link attached to it.
_DEFLECTION = re.compile(
    r"""(?ix)
    (?: (?:do(?:n't|es\ not|\ not)|did\ not)\s+ (?:have|contain|include|
                                                 mention|discuss|specify)
      | not\s+enough\s+(?:information|context|detail)
      | (?:no|nothing)\s+(?:clear\s+)?(?:discussion|mention|reference)\s+of
      | (?:is|are|was|were)\s+not\s+(?:discussed|mentioned|covered|addressed)
      | in\s+the\s+excerpts\s+provided
      | (?:i'?m|i\ am)\s+(?:here\ to|ready\ to|happy\ to)
      | could\ you\ (?:ask|clarify|be\ more)
      | can\ you\ (?:be\ more|clarify|tell\ me\ what)
      | (?:what|which)\ (?:specifically|exactly)\ (?:would|do)\ you
    )""")


HIGHLIGHTS = ROOT / "data" / "highlights.json"

# Openers for an unprompted fact, so twenty compliments do not produce
# twenty posts beginning the same way.
_HIGHLIGHT_LEADS = (
    "thanks 🙏 here's one people miss:",
    "appreciate it — one from the archive:",
    "🙏 here's a bit worth hearing:",
    "cheers. this one is worth a listen:",
    "thank you 🙏 one you might have skipped:",
)


def load_highlights(path: Path = HIGHLIGHTS) -> list[dict]:
    """Moments the bot can offer when nobody asked a question.

    Written ahead of time by scripts/make_highlights.py and read here,
    rather than generated per reply: a model call at reply time costs money,
    adds latency, and can produce a dud in public. A pool that was read
    before it went anywhere cannot.
    """
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("highlights.json is unreadable — compliments will "
                       "get silence rather than a bad fact")
        return []


def format_highlight(highlight: dict, seed: str) -> str:
    """A fact, its moment, and the episode it came from."""
    lead = _pick(_HIGHLIGHT_LEADS, seed)
    fact = soften(plain_text(strip_urls(highlight.get("text", ""))))
    stamp = highlight.get("timestamp", "")
    title = _fit(str(highlight.get("title", "")), 60)
    return f"{lead}\n\n{fact}\n\n{stamp} · {title}".strip()


def is_a_deflection(answer: str) -> bool:
    """Did the model decline rather than answer?

    Two signals, because the wording varies and the shape does not:

    A stock refusal phrase. And an answer that ends by asking the reader a
    question — a real answer to "what did X say" does not close with "could
    you ask about a specific moment?", and REPLY_STYLE already forbids it,
    which is exactly why a guard is needed rather than an instruction.
    """
    text = (answer or "").strip()
    if not text:
        return True
    if _DEFLECTION.search(text):
        return True
    tail = text.rstrip()[-160:]
    return tail.endswith("?")


def is_a_miss(answer: str) -> bool:
    """Did the model say it could not find this?

    The hit list cannot tell you. Retrieval always returns its top_k, so a
    question with no answer in the archive still comes back with six
    passages about something else — which is how the first real reply came
    out as "I couldn't find that" followed by a confident timestamp from an
    unrelated episode.
    """
    return NOT_FOUND_ANSWER.lower() in (answer or "").lower()


def _seconds(stamp: str) -> int:
    """"1:07:24" or "7:02" -> seconds."""
    parts = [int(p) for p in stamp.split(":")]
    total = 0
    for part in parts:
        total = total * 60 + part
    return total


def _relink(deep_link: str, seconds: int) -> str:
    """Point a seekable link at `seconds` instead of wherever it pointed.

    The link is built from the top retrieved passage, but the answer cites
    the line it actually used — often minutes away, and sometimes from a
    later passage entirely. Sending someone to the start of a passage while
    the text above says 1:07:24 is the same broken promise as citing the
    wrong time; the link should land where the words say it lands.
    """
    base = re.sub(r"[?&]t=\d+s?", "", deep_link or "")
    if not base or "youtube.com" not in base and "youtu.be" not in base:
        return deep_link
    joiner = "&" if "?" in base else "?"
    return f"{base}{joiner}t={seconds}s"


def weighted_length(text: str) -> int:
    """Length as X counts it: every URL is 23 characters."""
    total, urls = len(text), 0
    for word in text.split():
        if _URL_SHAPED.search(word):
            total -= len(word)
            urls += 1
    return total + urls * URL_WEIGHT


def wants_link(mode: str, deep_link: str) -> bool:
    """Should this particular reply carry its link?

    Three modes, because "links on" and "links off" are both wrong most of
    the time.

    A link costs $0.200 against $0.015 without one, whatever it points at.
    But the two kinds of link are not worth the same: a YouTube link carries
    ?t= and lands on the exact second, while an X broadcast link opens a
    four-hour video at 0:00 and leaves the reader to scrub. Paying thirteen
    times as much for the second one buys almost nothing — the episode name
    and the timestamp in the text get the reader to the same place.

    So "seekable" pays only when the link actually jumps. On this corpus
    that is about three answers in eight, which is roughly 60% off the link
    bill for no loss anyone would notice.
    """
    if mode == "always":
        return True
    if mode == "seekable":
        return "t=" in (deep_link or "")
    return False


def format_reply(answer: str, hits: list, include_links: bool | str = False,
                 limit: int = POST_LIMIT) -> str:
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
    answer = soften(plain_text(strip_urls(answer)))
    if is_a_miss(answer):
        # Just the sentence. The model tends to follow it with an offer to
        # try another question, which is fine on a web page and reads as
        # padding in a reply — and gets cut mid-word by the length budget.
        return _pick(_MISS_PHRASINGS, answer)
    if not hits:
        return _fit(answer, limit - 22)

    top = hits[0]
    mode = ("always" if include_links is True
            else "off" if include_links is False else str(include_links))
    if wants_link(mode, top.deep_link):
        seekable = "t=" in (top.deep_link or "")
        # Prefer the moment the answer actually names over the passage
        # start, and move the link to match it.
        cited = _CITES_A_TIME.search(answer)
        moment = cited.group(0) if cited else top.timestamp
        link = (_relink(top.deep_link, _seconds(moment)) if cited and seekable
                else top.deep_link)
        # Fitted first, because whether the tail should carry a timestamp
        # depends on whether the trimmed answer already has one — and the
        # tail's own length depends on that answer. Two passes, cheaply.
        # A seekable link earns a line saying so: the reader learns the
        # link jumps rather than just opens. An X link does not — it cannot
        # jump, and the answer has already named the moment, so a second
        # line repeating it said the same thing twice and left a dangling
        # dash where the card swallowed the URL.
        if seekable:
            lead = f"Jump to {moment}:"
        else:
            # X has no timestamp parameter for video, so this link opens at
            # 0:00 whatever the text says. Telling the reader to scrub is
            # the difference between a link that looks broken and one that
            # is honest about what it does — the website has said this for
            # months and the replies did not.
            lead = (f"Full episode — scrub to {moment} "
                    f"(X can't jump to a timestamp):")
        tail = f"\n\n{lead}\n{link}"
        return _fit(answer, limit - len(lead) - URL_WEIGHT - 3) + tail

    title = _fit(str(top.title), 60)
    # Assume the answer cites a moment, then check the TRIMMED text rather
    # than the original: deciding on the full answer and trimming afterwards
    # can cut the very timestamp that justified leaving it out.
    tail = f"\n\n{title}"
    fitted = _fit(answer, limit - 22 - len(tail))
    if not _CITES_A_TIME.search(fitted):
        tail = f"\n\n{top.timestamp} · {title}"
        fitted = _fit(answer, limit - 22 - len(tail))
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
    # Author ids that asked not to be contacted again. Permanent, and never
    # trimmed: honouring an opt-out for a while and then forgetting is worse
    # than never having offered one.
    opted_out: list = field(default_factory=list)
    # Indexes into the highlight pool that have already been offered, so the
    # account does not post the same fact twice.
    highlights_used: list = field(default_factory=list)
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
        """Persist what has been answered. Never fatal.

        This runs in a finally block, so an exception here does not fail a
        save — it fails the entire poll cycle, including replies that had
        already been posted. That is how a read-only data directory turned
        into a bot that answered nothing while reporting healthy.
        """
        try:
            self._save(path)
        except OSError as exc:
            logger.warning("could not persist state (%s) — continuing from "
                           "memory; a restart will re-seed from X", exc)

    def _save(self, path: Path) -> None:
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
            "opted_out": self.opted_out,
            "highlights_used": self.highlights_used[-200:],
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
                 post_limit: int = POST_LIMIT,
                 summaries=None, summary_limit: int = 4000,
                 highlights: list | None = None,
                 priority_authors: set | None = None,
                 site: str | None = None,
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
        self._post_limit = post_limit
        self._summaries = summaries
        self._summary_limit = summary_limit
        # Fetched once and kept: 32 summaries change only when an
        # episode is added, and a lookup should not cost a round trip.
        self._summary_cache: list | None = None
        # Injected rather than loaded here, so a caller — a test, or a
        # future surface with its own pool — can supply its own.
        self._highlights = (load_highlights() if highlights is None
                            else highlights)
        # Accounts that must never be met with silence — the hosts, the
        # show, the people who could actually put this in front of an
        # audience. A stranger getting no reply costs nothing. One of
        # them getting no reply is the only failure here that does.
        self._priority = set(priority_authors or ())
        self._site = site
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
            # Seed from X before deciding anything. The replied-set lives in
            # the same state file that a deploy wipes, so "answer anything
            # recent" meant re-answering questions that had already been
            # answered — three replies to one mention across three deploys.
            #
            # The account's own timeline cannot be lost, so it is the record
            # to trust here rather than local memory.
            try:
                already = await self._client.replied_to()
            except Exception:                                  # noqa: BLE001
                logger.exception("could not seed from X — skipping the "
                                 "backlog rather than risk repeating it")
                already = None
            if already is None:
                self.state.since_id = mentions[-1].id
                self.state.save(self._state_path)
                return 0
            if already:
                self.state.replied = sorted(already)[-500:]
                logger.info("seeded %d already-answered mention(s) from X",
                            len(already))

            # Cold start. Render's disk is ephemeral, so this happens on
            # every deploy — not only the first ever run.
            #
            # Skipping everything was safe and wrong. Replying to a month of
            # backlog at once is the pattern that gets accounts suspended,
            # but a question asked two minutes before a deploy is not
            # backlog, and dropping it silently is exactly what makes the
            # account look broken. That happened: five deploys in an hour
            # ate the same question twice while someone was watching.
            #
            # So the line is time, not existence. Anything newer than
            # COLD_START_GRACE still gets answered; older stays skipped.
            recent = [m for m in mentions if _is_recent(m.created_at)]
            self.state.since_id = mentions[-1].id
            if recent:
                self.state.since_id = _before(recent[0].id)
                logger.info("cold start — skipping %d old, answering %d "
                            "from the last %d minutes",
                            len(mentions) - len(recent), len(recent),
                            COLD_START_GRACE // 60)
                self.state.save(self._state_path)
                return await self._tick(today)
            logger.info("cold start — skipping %d existing mention(s)",
                        len(mentions))
            return 0

        posted = 0
        replied = set(self.state.replied)
        opted_out = set(self.state.opted_out)
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
            if mention.author_id in opted_out:
                # Permanent. Checked before anything else that could produce
                # a reply, because the promise made in the opt-out is that
                # this account is never contacted again.
                handled = mention.id
                continue
            if asks_to_be_left_alone(mention.text):
                logger.info("%s asked to opt out — honouring it permanently",
                            mention.author_id)
                opted_out.add(mention.author_id)
                self.state.opted_out.append(mention.author_id)
                handled = mention.id
                continue
            if self._verified_only and not mention.author_verified:
                # Checked here rather than inside compose(), so an ignored
                # account costs nothing beyond the read that already
                # happened — no retrieval, no model call, no reply.
                logger.info("%s is from an unverified account — skipping",
                            mention.id)
                handled = mention.id
                continue
            if (self.state.replies_today + posted >= self.cap
                    and mention.author_id not in self._priority):
                # A priority account is answered even on a day the cap has
                # already been reached: the cap exists to bound a stranger's
                # spam, and these are the people it must never silence.
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

    def _fallback(self, mention: Mention) -> str | None:
        """Something to say when the answer was not good enough to post.

        Only for the priority accounts. Everyone else gets silence, which is
        correct — a weak reply to a stranger is worse than none. But silence
        aimed at one of the hosts reads as a broken tool in front of exactly
        the people who would otherwise pass it on, so they get a real fact
        from the archive rather than nothing.
        """
        found = self._next_highlight(mention.id)
        if found:
            logger.info("%s is a priority account — offering a fact instead "
                        "of silence", mention.author_id)
            return format_highlight(found, mention.id)
        return None

    def _next_highlight(self, seed: str) -> dict | None:
        """A moment this account has not offered before.

        Repeats are the thing to avoid — posting the same fact twice is the
        duplicative-content problem in a different costume — so used ones
        are remembered, and the pool reshuffles only once every one has been
        spent.
        """
        if not self._highlights:
            return None
        used = set(self.state.highlights_used)
        fresh = [h for i, h in enumerate(self._highlights) if i not in used]
        if not fresh:
            self.state.highlights_used = []
            fresh = list(self._highlights)
        chosen = fresh[int(hashlib.sha256(seed.encode()).hexdigest(), 16)
                       % len(fresh)]
        self.state.highlights_used.append(self._highlights.index(chosen))
        return chosen

    async def _summary_for(self, number: int) -> dict | None:
        """The stored summary for an episode number, if there is one.

        Where a show exists as both a YouTube cut and a live broadcast, the
        longer one wins: it is the version that actually contains
        everything, and the summary of a cut is a summary of a cut.
        """
        if self._summaries is None:
            return None
        if self._summary_cache is None:
            self._summary_cache = await self._summaries.list_all()
        matches = [s for s in self._summary_cache
                   if episode_number(s.get("title", "")) == number]
        if not matches:
            return None
        return max(matches, key=lambda s: len(s.get("summary", "")))

    async def compose(self, mention: Mention) -> str | None:
        """The reply this mention would get, or None to stay quiet.

        Separate from posting so a reply can be read before it is sent.
        Every reply defect found so far came from looking at composed output
        on a real question rather than from a test.
        """
        question = question_from(mention.text)
        # A bare tag with nothing attached gets nothing back. Anything else
        # falls through: the length check exists to keep retrieval from
        # running on nothing, not to decide who deserves a reply, and it was
        # silencing "lfg" and "gm" before the highlight path was reached.
        if not question:
            logger.info("%s is a bare tag — skipping", mention.id)
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

        about = about_answer(question, self._site)
        if about:
            logger.info("%s asked what this is — answering from the fixed "
                        "description", mention.id)
            return about

        wanted = summary_request(question)
        if wanted is not None:
            found = await self._summary_for(wanted)
            if found:
                mode = ("always" if self.include_links is True
                        else "off" if self.include_links is False
                        else str(self.include_links))
                return format_summary(
                    found["summary"], found["title"], self._summary_limit,
                    url=found.get("url") if mode != "off" else None)
            logger.info("%s asked for episode %d, which is not indexed",
                        mention.id, wanted)
            return f"I don't have episode {wanted} indexed."

        if (len(question) < self._min_question
                or not looks_like_a_question(question)):
            # Not a question, so retrieval would have nothing to work with.
            # But silence in front of someone who just said something nice
            # is a wasted moment: they are looking at the account, and what
            # would convince them is a demonstration rather than a
            # thank-you. So it offers a fact instead — one that was written
            # and read before it ever went anywhere near a reply.
            found = self._next_highlight(mention.id)
            if found:
                logger.info("%s is not a question — offering a highlight",
                            mention.id)
                return format_highlight(found, mention.id)
            logger.info("%s is not a question (%r) — staying quiet",
                        mention.id, question[:60])
            return None

        priority = mention.author_id in self._priority
        result = await self._index.search(
            question, instruction=reply_style(self._post_limit))

        # Ask once more before giving up. The same question has produced a
        # flat "I couldn't find that" one minute and a good cited answer the
        # next, from the same passages — the model simply gives up sometimes.
        # A miss is the reply people screenshot as proof it does not work, so
        # it is worth $0.008 to be sure, and only when retrieval actually
        # found something to work with.
        if is_a_miss(result.answer) and len(result.hits) >= 3:
            logger.info("%s missed on the first pass — asking again",
                        mention.id)
            retry = await self._index.search(
                question, instruction=reply_style(self._post_limit))
            if not is_a_miss(retry.answer):
                result = retry
        if getattr(result, "refused", False):
            logger.info("%s refused by the model — staying quiet", mention.id)
            return self._fallback(mention) if priority else None
        if is_a_deflection(result.answer):
            # A non-answer with a timestamp in it still passes the citation
            # check, which is how "I don't have enough information" reached
            # a live reply with a link attached.
            logger.info("%s deflected (%r) — staying quiet",
                        mention.id, result.answer[:70])
            return self._fallback(mention) if priority else None
        if not is_a_miss(result.answer) and not _CITES_A_TIME.search(result.answer):
            # A real answer from this index always names a moment — the
            # prompt requires it, and citing is the entire point. An answer
            # with no timestamp that is not the honest "couldn't find it" is
            # the model talking about itself ("I appreciate your enthusiasm,
            # but I'm here to answer questions about..."), which reached a
            # reply once with an unrelated episode stapled underneath.
            logger.info("%s produced no citation (%r) — staying quiet",
                        mention.id, result.answer[:70])
            return self._fallback(mention) if priority else None
        return format_reply(result.answer, result.hits,
                            include_links=self.include_links,
                            limit=self._post_limit) or None

    async def _answer(self, mention: Mention) -> bool:
        text = await self.compose(mention)
        if not text:
            return False
        # Any URL still in the text at this point is one this code put
        # there — a deep link, or the site in the "what is this" answer.
        # Transcript URLs were stripped much earlier, which is what the
        # guard in reply() is actually for. Deriving the flag from the
        # include_links setting instead meant the about answer, which
        # carries the site link by design, could not be posted at all when
        # links were off.
        posted = await self._client.reply(
            text, mention.id, allow_link=bool(_URL_SHAPED.search(text)))
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
