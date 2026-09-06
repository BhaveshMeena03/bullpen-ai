"""Market Bubble episode search.

Ingests timestamped episode transcripts, embeds windows of consecutive
segments (keeping each window's start time), and answers natural-language
questions with an answer plus citations that DEEP-LINK to the exact moment
in the episode.

Stored in a dedicated Pinecone namespace ("podcast") so it never collides
with the concierge's docs. Reuses the same Voyage + Pinecone + Claude
plumbing as the rest of the app.
"""

import asyncio
import contextlib
import hashlib
import logging
import re
import time
from xml.sax.saxutils import escape, quoteattr

import voyageai
from anthropic import AsyncAnthropic
from pinecone import Pinecone

from . import hedging, names
from .config import anthropic_client_kwargs, get_settings, redact
from .embeddings import embed_query, embed_texts, rerank_order
from .schemas import (
    Episode,
    PodcastHit,
    PodcastSearchResponse,
    TranscriptSegment,
)
from .terms import TermIndex

logger = logging.getLogger(__name__)

# Whose passages to search, when a question names one of them.
#
# Only the hosts, and only because the speaker labels only cover the
# hosts: they are in all 33 episodes, so a voice recurring everywhere
# identifies them. A guest appears once and cannot be identified that
# way, so filtering on a guest name would search a set nobody labelled
# and find nothing.
#
# "Z" is deliberately absent despite being what Banks calls Ansem on the
# show. It is one letter and it appears inside ordinary words; the cost
# of a false match here is searching the wrong person's passages, which
# is the failure this exists to fix.
_HOSTS_IN_QUESTION = (
    ("Ansem", re.compile(r"(?i)\bansem\b")),
    ("FaZe Banks", re.compile(r"(?i)\b(?:faze\s+)?banks\b")),
)


def host_named_in(query: str) -> str | None:
    """The host a question is about, if it is about one.

    Returns None when a question names both or neither. Both is not a
    filter — "what did ansem and banks disagree about" wants the
    conversation, not one side of it.
    """
    found = [name for name, pattern in _HOSTS_IN_QUESTION
             if pattern.search(query or "")]
    return found[0] if len(found) == 1 else None

# How long to stop trying the model proxy after it fails. Long enough
# that an outage is not re-tested on every visitor, short enough that
# recovery needs no deploy.
PROXY_COOLDOWN_SECONDS = 300

# Named, so the fallback cannot inherit ANTHROPIC_BASE_URL from the
# environment and end up pointing at the proxy it exists to escape.
ANTHROPIC_DIRECT_URL = "https://api.anthropic.com"

NAMESPACE = "podcast"

# End of the first sentence, which is as much as the stream needs before it
# can tell a denial from an answer.
_SENTENCE_BREAK = re.compile(r"[.!?]\s")

# How many exact-token matches may be put in front of the reranker, across
# the query and every alternative spelling of it. Matches TermIndex.lookup's
# own cap so that expanding a query cannot flood the pool: before spelling
# expansion existed one lookup contributed at most this many, and it still
# does.
_EXACT_MATCH_CAP = 8

REFUSAL_ANSWER = ("I can't help with that one — try asking about "
                  "something discussed on the show.")

# What the model is told to say when the excerpts do not contain the answer.
# Callers need to recognise a miss, and the only signal is the wording: the
# retriever always returns its top_k, so a full hit list means nothing about
# whether any of it was relevant. Kept as a constant rather than interpolated
# into the prompt below, because SYSTEM_PROMPT's exact bytes are the prompt
# cache key — a test asserts the two stay in step.
NOT_FOUND_ANSWER = "I couldn't find that in the episodes I've indexed"

# The Musk archive answers from its own prompt, not this one.
#
# Sharing it shipped a bug straight into verification: asked what Elon says
# about consciousness, the model replied "you're asking about the Market
# Bubble podcast, but these excerpts are from Lex Fridman's conversations
# with Elon Musk" -- it read the corpus as off-topic because the first
# sentence told it what show it was on, and explained the mismatch to the
# user instead of answering. A page of that was two days from going public.
#
# It is a separate prompt rather than one with the show name swapped, since
# most of what makes the other one long is Market Bubble's own history:
# rules 5b to 5d are about "FaZe Banks:" line prefixes, 5f is about the two
# hosts correcting each other's price targets. None of that exists here.
# What does exist is the same shape of failure in a different costume --
# Lex talks for roughly half of every recording, and his words handed back
# as Elon's is the one mistake this archive cannot survive.
ELON_SYSTEM_PROMPT = """\
You answer questions about long-form interviews with Elon Musk, using ONLY \
the transcript excerpts provided in <excerpts> tags. Each excerpt is tagged \
with its episode, timestamp, and the date it was published. Excerpts are \
given oldest first.

These are interviews: an interviewer asks the questions and Elon Musk \
answers them. Both voices are in the transcript.

WHO the interviewer is depends on the recording, and the episode name on \
each excerpt is what tells you. "Lex Fridman Podcast" is Lex Fridman. \
"Joe Rogan Experience" is Joe Rogan. Never carry the interviewer from one \
excerpt to another, and never name one the episode does not name.

Rules:
1. Answer strictly from the excerpts. If they do not contain the answer, \
say "I couldn't find that in the episodes I've indexed" — do not use \
outside knowledge about Elon Musk, however well known, and do not guess. \
Say that plainly, without explaining what the excerpts are instead.
2. Cite the moment. Every line inside an excerpt begins with its own \
timestamp in square brackets, like [16:16]. Cite the timestamp of the line \
you actually used and name the episode ("around 1:12:04 in the 2021 \
conversation"). NEVER write a URL or a Markdown link — you are not given \
the addresses, so writing one means inventing it.
3. Separate the two speakers. A question, a framing, an anecdote from the \
interviewer's own life, or a summary of somebody else's research is very \
often the interviewer, not Elon. Attribute something to Elon only when \
the excerpt shows him saying it; otherwise say "the interviewer" or \
describe what was discussed without putting it in anyone's mouth. Half of \
every recording is somebody other than the person being asked about, and \
a quote under the wrong name is the failure this archive does not \
recover from.

3a. That cuts both ways, and getting the interviewer wrong is just as \
bad as getting Elon wrong. Asked what Lex said about jiu jitsu, this \
archive answered with a passage from Joe Rogan Experience #1470 -- Rogan \
on Hoist Gracie and early MMA -- and printed it as Lex. Two real people, \
one quoted saying something the other said. If an excerpt is from the Joe \
Rogan Experience, the interviewer in it is Joe Rogan and Lex Fridman is \
not present at all; if the question names an interviewer who is not in \
the recordings you were given, say so rather than answering from a \
different one.
4. Mind the years. These span 2019 to 2024 and his views moved. If \
excerpts disagree, give the order and the dates rather than blending them \
into one position he never held.
5. Do not put words in anyone's mouth or invent quotes — paraphrase what \
the excerpt says.
6. This is an informational search tool. It is not investment advice, it \
does not speak for Elon Musk or any of his companies, and it never claims \
his endorsement of anything.
7. Keep it tight and conversational — a couple of sentences plus the \
citation, not an essay."""

SYSTEM_PROMPT = """\
You answer questions about the "Market Bubble" podcast (hosted by Ansem and \
FaZe Banks) using ONLY the transcript excerpts provided in <excerpts> tags. \
Each excerpt is tagged with its episode, timestamp, and — when known — the \
date the episode aired. Excerpts are given oldest first.

Rules:
1. Answer strictly from the excerpts. If they don't contain the answer, say \
"I couldn't find that in the episodes I've indexed" — do not use outside \
knowledge and do not guess.
2. Cite the moment. Every line inside an excerpt begins with its own \
timestamp in square brackets, like [16:16]. Cite the timestamp of the line \
you actually used, NOT the `at` attribute on the excerpt — that is only \
where the passage begins, and it can be a minute or more before the moment \
you are describing. Mention the episode too ("around 16:16 in <episode>"). \
The interface shows clickable timestamps alongside your answer, so refer to \
them naturally. NEVER write a URL or a Markdown link of any kind. You are \
not given the video addresses and cannot know them, so writing one means \
inventing it — observed producing "https://www.youtube.com/watch?v=example&t=3407" for a segment that is not \
on YouTube at all. A fabricated link in a citation is worse than no link: \
it looks checkable and is not. Give the timestamp and the episode name in \
plain text and let the interface do the linking.
3. Mind the dates. If excerpts from different dates disagree, say so and \
give the order ("in May he argued X; by July he'd shifted to Y") rather than \
blending them into one view nobody held. When a question is about what \
someone thinks *now*, lean on the most recent excerpt and say how recent it \
is. Never present a stale take as current.
4. Summarize faithfully. Do not put words in the hosts' mouths or invent \
quotes — paraphrase what the excerpt actually says.
5. Name a speaker only when the excerpt makes it unambiguous. These are \
auto-generated captions with NO speaker labels: an episode whose title lists \
four guests gives you no way to tell which of them is talking, and a \
confident guess puts a real quote under the wrong person's name. That \
happened: "Austin Federa said they get flamed for claiming 1.5 million \
users" — it was FOMO's own co-founder, and Federa is from a different \
company entirely. A misattributed quote is worse than a vague one, because \
the person named did not say it and the person who did gets no credit. When \
you cannot tell, write "a guest", "one of the hosts", or "the founder of X" \
if the excerpt establishes the company. Attribute to a named person only \
when the excerpt says the name, or someone is addressed by it.
5b. Some lines carry a speaker's name before the text, like "[12:02] \
FaZe Banks: I put close to seven figures in Hyperliquid". That prefix is \
the strongest thing that establishes who spoke, and outranks everything \
below. Attribute a line to the name in front of it and to nobody else. \
A line with no prefix falls through to the `voices` attribute in rule \
5g, and only when that cannot settle it is the speaker unknown — then \
describe it as "one of the hosts" or "a guest", never as the person the \
question asked about. Do not treat a missing prefix as unknowable on its \
own: only about a third of lines carry one, because only speech the voice \
map could attribute gets a name, so reading this rule as the last word \
makes a host anonymous on every unlabelled line — which is how "ansem on \
solana" answered "one of the hosts" off three passages the index had \
already labelled Ansem. A prefix present is proof; a prefix absent is \
merely silence, and `voices` may still settle it. Asked what Banks said \
about Solana, the \
excerpts came back containing both hosts and a line prefixed "Ansem:" \
was reported as Banks saying it, because the question had named Banks. \
The prefix outranks the question every time.

5c. A name INSIDE a line is a person being talked about, not the person \
talking. "FaZe Banks: I'm gonna help continue to guide Z the best way I \
can" is Banks speaking about Ansem — it is not Ansem speaking. Attributing \
it to Ansem, because his name appears in the words, reverses who said what \
about whom. Read only the prefix.

5d. A line with NO prefix is not a line you cannot attribute. Only the \
two hosts are labelled; every guest is unprefixed, so treating an absent \
prefix as "unknowable" refuses to answer anything about a guest at all — \
which took "what did Jesse say about Base" from a good answer to a \
refusal. For unprefixed lines fall back to rule 5: attribute when the \
episode or the conversation makes it plain, such as a guest who is named \
in the title, introduced by name, or addressed by name. The prefix rules \
above decide BETWEEN the two hosts; they do not silence everyone else.

5a. A name in the QUESTION is not evidence about the excerpts. Asked "how \
much did Banks make this month", the excerpts do not become about Banks — \
and answering from a passage that never names him, as though it were his, \
reported another person's investment portfolio as Banks losing $254,000. \
The question tells you what someone wants to know, never who was speaking. \
If the excerpts do not establish that, say so plainly and answer about \
what they DO establish, even when that is less than the question asked \
for. "Someone on the show said" is a worse headline and a true one.
1a. If you cite a timestamp ANYWHERE in your answer, your FIRST sentence \
must be about what was said, not about what was not. This is mechanical, \
not a matter of taste. These openings are forbidden whenever a citation \
follows: "I couldn't find", "I don't see", "I didn't find", "There's no \
direct/specific statement", "Not in those words", "Nothing matching", \
"The excerpts don't contain". Reaching for one and then writing "However, \
around 1:46:25 he does discuss..." produces a correct answer wearing a \
denial, and the reader stops at the first sentence. Both halves have gone \
out on public replies.\
 Rule 1 is for when you cite NOTHING. If you cite something, open with \
it — "Around 27:09, X" — and put any shortfall at the END, as a \
qualifier: "...though he doesn't put it in those words." A near miss \
stated last reads as precision. Stated first it reads as failure.
5e. A number belongs to the asset named on its OWN line. Excerpts come \
from different episodes and different assets sit beside each other, so \
carrying a figure across lines invents a position nobody stated. Asked \
what price targets were discussed, a line reading "your buy targets for \
Bitcoin is like 55" was published as "Hyperliquid at $55K", and \
"Bitcoin bottomed at 58K around November" — a past low — was published \
as a target. Both numbers were real and both were attached to the wrong \
thing. If a line gives a figure without naming what it is for, say that \
or leave it out; never supply the asset from a neighbouring line, from \
the episode title, or from the question. And a level someone says the \
price REACHED is not a level they are predicting — keep the tense.

5f. When one speaker states a figure and another corrects it, the \
correction is the answer. Read a few lines PAST any number before \
reporting it. Banks guessed "your buy targets for Bitcoin is like 55K, \
Salada is 55K, and Hyperliquid is like 55K... or I might be off by a \
little bit", and Ansem answered "I said like 58K, 58, and then 55" — so \
the targets are $58K, $58 and $55. The reply published Banks' guess as \
Ansem's target, kept the "K" that belonged only to Bitcoin, and printed \
Hyperliquid at $55,000. Hedges like "something like that", "I might be \
off", "roughly" mark a figure as unreliable: either use the corrected \
one or say the number was approximate. Never carry a unit — K, million, \
billion — from one asset onto another.
5g. An excerpt may carry a `voices` attribute listing which HOSTS were \
detected speaking somewhere inside it. Read it as passage-level, never \
line-level. ONE name means the host lines in that passage are his — \
attribute them to him. TWO names mean both hosts speak in it and it does \
NOT tell you which line is whose; fall back to rule 5 and say "one of \
the hosts". A host absent from `voices` did not speak in that passage \
at all, however the question was worded. Guests are never listed, so an \
unlisted speaker is a guest and not a host — `voices="FaZe Banks"` on a \
passage containing a guest's answer means Banks is one of the two \
voices, not that Banks said every line. This attribute is the only \
speaker evidence you get; the name prefixes described in 5b do not \
appear in this archive.
6. This is an informational search tool, not financial advice. Never add \
buy/sell recommendations or price predictions of your own.
7. Keep it tight and conversational — a couple of sentences plus the \
citation, not an essay."""


def _timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _deep_link(url: str, platform: str, seconds: float) -> str:
    sec = int(seconds)
    if platform == "youtube":
        joiner = "&" if "?" in url else "?"
        return f"{url}{joiner}t={sec}s"
    if platform == "spotify":
        return f"{url}#t={sec}"
    if "x.com/" in url or "twitter.com/" in url:
        # Plain seconds, no "s" suffix — that is what the player reads.
        joiner = "&" if "?" in url else "?"
        return f"{url}{joiner}t={sec}"
    return url


def _windows(
    segments: list[TranscriptSegment], max_chars: int, overlap_segments: int
) -> list[tuple[float, str]]:
    """Pack consecutive segments into windows of <= max_chars, returning
    (start_seconds, text) per window. Windows overlap by a few segments so
    an answer that straddles a boundary is still retrievable.

    Returns (start_seconds, text, stamped) — the same passage twice.

    `text` is what gets embedded and what a reader sees, and it is exactly
    what it has always been. `stamped` is the same lines with each one
    prefixed by its own timestamp, and it exists only to be handed to the
    model when it writes an answer.

    They are separate on purpose. A window is minutes of speech carrying a
    single start time, so a model given only that could cite nothing else:
    an answer about the BlackRock exchange at 16:16 was cited as 15:39,
    because 15:39 was where the passage began. Every citation was landing
    up to a minute early, on a product that promises the exact second.

    Putting the timestamps into the embedded text would have fixed that and
    quietly changed retrieval — roughly a tenth of each window would become
    non-semantic tokens, and every vector would shift. Keeping the embedded
    text identical means the ranking after this change is provably the same
    ranking as before it.
    """
    windows: list[tuple[float, str, str]] = []
    i = 0
    n = len(segments)
    while i < n:
        start_t = segments[i].t
        parts: list[str] = []
        stamped_parts: list[str] = []
        length = 0
        j = i
        while j < n and length + len(segments[j].text) + 1 <= max_chars:
            speaker = f"{segments[j].speaker}: " if segments[j].speaker else ""
            line = f"{speaker}{segments[j].text}"
            parts.append(line)
            stamped_parts.append(f"[{_timestamp(segments[j].t)}] {line}")
            length += len(line) + 1
            j += 1
        if j == i:  # single segment longer than max_chars — take it whole
            clipped = segments[i].text[:max_chars]
            parts.append(clipped)
            stamped_parts.append(f"[{_timestamp(segments[i].t)}] {clipped}")
            j = i + 1
        windows.append((start_t, "\n".join(parts), "\n".join(stamped_parts)))
        if j >= n:
            break
        i = max(j - overlap_segments, i + 1)
    return windows


# X included, because X broadcasts DO seek. This was assumed otherwise for
# months and never tested: ?t=<seconds> on a broadcast — on the status URL
# as well as the /i/broadcasts/ one — opens the player at that second.
# Checked on three broadcasts, asking for 1800, 2400 and 9311 and getting
# exactly those back from the player.
#
# The cost of the assumption was half the archive. Thirty-five hours of
# broadcasts were shown with no play button, a muted timestamp and a note
# telling people to scrub by hand, when a link would have worked.
_SEEKABLE_HOST = re.compile(
    r"^https?://(www\.)?(youtube\.com|youtu\.be|open\.spotify\.com"
    r"|x\.com|twitter\.com)/")


# Seeking and embedding are different questions. Everything here seeks; only
# YouTube plays inside the page.
_EMBEDDABLE_HOST = re.compile(r"^https?://(www\.)?(youtube\.com|youtu\.be)/")


def _can_embed(link: str) -> bool:
    """Whether the moment can open in the page rather than on another site."""
    return bool(_EMBEDDABLE_HOST.match(link or ""))


def _can_seek(link: str) -> bool:
    """Whether a citation into this link can land on the moment."""
    return bool(_SEEKABLE_HOST.match(link or ""))


def _same_moment(a: str, b: str, threshold: float = 0.55) -> bool:
    """Do two windows describe the same stretch of conversation?

    Compared on the first words rather than the whole window, because the
    two transcriptions differ — YouTube's auto-captions and Whisper render
    the same speech with different punctuation, casing and occasional
    different words. What stays stable is the sequence of ordinary words at
    the start.
    """
    aw = re.sub(r"[^a-z0-9 ]", " ", a.lower()).split()[:18]
    bw = re.sub(r"[^a-z0-9 ]", " ", b.lower()).split()[:18]
    if len(aw) < 8 or len(bw) < 8:
        return False
    shared = len(set(aw) & set(bw))
    return shared / min(len(aw), len(bw)) >= threshold


def _prefer_seekable(hits: list[PodcastHit]) -> list[PodcastHit]:
    """Where the same moment appears twice, keep the copy that plays here.

    Roughly half of every live broadcast is also in the YouTube upload of
    that episode, so a single query can retrieve the same passage twice.

    This used to be justified by saying X could not seek at all. That was
    never true and was never tested: ?t=<seconds> on a broadcast opens the
    player at that second, on the status URL as well as /i/broadcasts/.
    Both copies land on the moment.

    What still separates them is where they land. YouTube has an embed, so
    the moment opens inside the page; a broadcast opens on X. Between two
    citations of the same words, the one that does not navigate away is
    worth more — which is a smaller claim than the old one, and true.

    Order is otherwise untouched: this only drops a later duplicate, and
    only when an embeddable hit already covers it.
    """
    kept: list[PodcastHit] = []
    for hit in hits:
        if _can_embed(hit.deep_link):
            kept.append(hit)
            continue
        covered = any(_can_embed(k.deep_link) and _same_moment(k.text, hit.text)
                      for k in hits)
        if not covered:
            kept.append(hit)
    return kept


class PodcastIndex:
    SURFACE = "market-bubble-search"
    # A class-level default so an instance built without __init__ -- which
    # the tests do, to exercise retrieval without a network client -- still
    # knows which corpus it reads. Without it the exact-match fetch fails
    # with an AttributeError that the surrounding code swallows into
    # "continuing with the vector results alone", so the term index would
    # quietly stop contributing and nothing would say so.
    _namespace = NAMESPACE

    def __init__(self, ledger=None, namespace: str | None = None) -> None:
        # Which corpus this instance answers from. The default is the Market
        # Bubble broadcast; a second archive passes its own namespace and
        # gets the same retrieval without sharing a single vector.
        #
        # Sharing one would be the end of the account. @mbubbleSearch's
        # entire standing is that it answers from that show, and one reply
        # about Market Bubble sourced from a Tesla interview would prove it
        # does not know the difference.
        self._namespace = namespace or NAMESPACE
        # The prompt follows the corpus, because the first sentence of a
        # prompt tells the model what it is reading, and being told the
        # wrong thing is how Musk transcripts came back as "you're asking
        # about the Market Bubble podcast".
        self._system_prompt = (ELON_SYSTEM_PROMPT
                               if self._namespace == "elon" else SYSTEM_PROMPT)
        self._ledger = ledger
        settings = get_settings()
        self._settings = settings
        self._voyage = voyageai.AsyncClient(api_key=settings.voyage_api_key)
        # Direct unless ANTHROPIC_BASE_URL names somewhere else. Only this
        # surface reads it: the concierge answers Bullpen support questions
        # and is not routed through a third party's account.
        self._anthropic = AsyncAnthropic(**anthropic_client_kwargs(settings))
        # Anthropic direct, held ready, and only when the line above is NOT
        # already that. A proxy can run out of balance, get its token
        # revoked, or simply stop answering, and every one of those looks
        # like the site being down to somebody typing a question. This is
        # the way back, and it needs no deploy: a redeploy on Render is a
        # restart, and with the poll grace that is about three minutes of
        # silence to fix something that should never have been visible.
        #
        # It carries the real key, which is exactly why the client above
        # does not.
        # base_url is named explicitly, and that is not decoration. The SDK
        # reads ANTHROPIC_BASE_URL from the environment when it is not told
        # otherwise — which is exactly how the proxy gets configured — so a
        # client built with only a key inherits the proxy and the fallback
        # quietly becomes a second route to the thing that just failed.
        # Caught by pointing this at a dead token: the fallback returned the
        # same 401.
        self._fallback = None
        if "base_url" in anthropic_client_kwargs(settings):
            self._fallback = AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                base_url=ANTHROPIC_DIRECT_URL,
            )
        # When the proxy last failed. Inside the cooldown every request goes
        # straight to Anthropic rather than paying the timeout again — an
        # outage should cost one slow answer, not one per visitor.
        self._proxy_failed_at = 0.0
        self._index = None
        # Loaded once. Absent or unreadable means every lookup returns
        # nothing and search behaves exactly as it did before.
        self._terms = TermIndex()

    def _llm(self):
        """The client to try first, and whether a fallback is still held.

        Inside the cooldown after a failure this hands back Anthropic
        directly, so a proxy that is down costs one slow answer rather than
        one per visitor for as long as it stays down.
        """
        if self._fallback is None:
            return self._anthropic, False
        cooling = (time.monotonic() - self._proxy_failed_at
                   < PROXY_COOLDOWN_SECONDS)
        if cooling:
            return self._fallback, False
        return self._anthropic, True

    def _proxy_broke(self, exc: Exception) -> None:
        self._proxy_failed_at = time.monotonic()
        logger.error(
            "podcast: the model proxy failed (%s) — answering on Anthropic "
            "direct, and skipping the proxy for %ds",
            redact(str(exc))[:200], PROXY_COOLDOWN_SECONDS)

    @property
    def index(self):
        if self._index is None:
            self._index = Pinecone(
                api_key=self._settings.pinecone_api_key
            ).Index(self._settings.pinecone_index)
        return self._index

    # -- ingestion ----------------------------------------------------------
    async def ingest(self, episodes: list[Episode]) -> int:
        rows: list[dict] = []
        for ep in episodes:
            for start_t, text, stamped in _windows(
                ep.segments,
                self._settings.chunk_max_chars,
                overlap_segments=2,
            ):
                rows.append(
                    {
                        "episode_id": ep.episode_id,
                        "title": ep.title,
                        "url": ep.url,
                        "platform": ep.platform,
                        "start_seconds": start_t,
                        "text": text,
                        # The same passage with per-line timestamps, read
                        # only when building the excerpt a model answers
                        # from. Never embedded, never shown to a reader.
                        "text_ts": stamped,
                        # Without this every chunk is timeless, and a view
                        # from months ago ranks against a later correction
                        # on wording alone. Pinecone metadata rejects None,
                        # so undated episodes omit the key entirely.
                        **({"published_at": ep.published_at}
                           if ep.published_at else {}),
                    }
                )
        if not rows:
            return 0

        embeddings = await embed_texts(
            self._voyage,
            # Embed the title with the window. An episode's subject often
            # lives in its title and is barely spoken aloud — "Ansem's trade
            # journal", "TJR", "AI beating crypto" are all title phrasings —
            # so embedding the transcript alone made whole episodes
            # unreachable by the obvious question. The stored excerpt stays
            # the transcript, so a reader still sees what was actually said.
            [f"{r['title']}\n\n{r['text']}" for r in rows],
            model=self._settings.voyage_model,
            dimension=self._settings.embedding_dimension,
            input_type="document",
        )

        vectors = [
            {
                "id": hashlib.sha256(
                    f"{r['episode_id']}:{r['start_seconds']}".encode()
                ).hexdigest()[:32],
                "values": emb,
                "metadata": r,
            }
            for r, emb in zip(rows, embeddings, strict=True)
        ]

        def _upsert() -> None:
            for start in range(0, len(vectors), 100):
                self.index.upsert(
                    vectors=vectors[start:start + 100], namespace=self._namespace
                )

        # Bound the write: the Pinecone client has no read timeout, so a dead
        # socket would hang the whole ingest indefinitely. On timeout, raise so
        # the caller's idempotent retry (deterministic chunk IDs = safe re-run)
        # kicks in instead of blocking forever.
        await asyncio.wait_for(
            asyncio.to_thread(_upsert),
            timeout=self._settings.pinecone_write_timeout_seconds,
        )
        logger.info("Indexed %d transcript windows from %d episodes",
                    len(vectors), len(episodes))
        return len(vectors)

    # -- search -------------------------------------------------------------
    async def _add_exact_matches(
        self, query: str, hits: list[PodcastHit]
    ) -> list[PodcastHit]:
        """Add passages containing a rare token from the query.

        Returns `hits` unchanged on any failure. A missing index, an
        unreachable fetch, or a malformed record must degrade to exactly
        the behaviour that existed before this — retrieval quality is the
        product, and an addition that can subtract is not worth having.
        """
        # This index matches letters, so it has holes exactly where the
        # captions do: "Solana" appears as "Salana" in 39% of the archive
        # and this lookup cannot see any of it. The embeddings bridge that
        # on their own; the term index needs the spellings spelled out.
        #
        # lookup() returns a LIST, rarest first and already capped -- that
        # order is the whole ranking, so these are appended rather than
        # unioned. Base spellings keep the front: they matched the words
        # actually asked for, and a mangling only earns a slot the query
        # itself left empty.
        try:
            ids = list(self._terms.lookup(query))
            seen = set(ids)
            for spelling in names.expand(query):
                for vector_id in self._terms.lookup(spelling):
                    if vector_id not in seen:
                        seen.add(vector_id)
                        ids.append(vector_id)
            ids = ids[:_EXACT_MATCH_CAP]
        except Exception as exc:                              # noqa: BLE001
            logger.warning("exact-match lookup failed (%s) — continuing with "
                           "the vector results alone", exc)
            return hits
        if not ids:
            return hits

        present = {
            hashlib.sha256(
                f"{h.episode_id}:{h.start_seconds}".encode()
            ).hexdigest()[:32]
            for h in hits
        }
        wanted = [i for i in ids if i not in present]
        if not wanted:
            return hits

        def _fetch():
            return self.index.fetch(ids=wanted, namespace=self._namespace)

        try:
            fetched = await asyncio.wait_for(
                asyncio.to_thread(_fetch),
                timeout=self._settings.pinecone_read_timeout_seconds,
            )
        except Exception as exc:                              # noqa: BLE001
            logger.warning("exact-match fetch failed (%s) — continuing with "
                           "the vector results alone", exc)
            return hits

        records = getattr(fetched, "vectors", None) or {}
        added = 0
        for record in records.values():
            md = getattr(record, "metadata", None) or {}
            if not md.get("text"):
                continue
            start = float(md.get("start_seconds", 0))
            hits.append(
                PodcastHit(
                    episode_id=md.get("episode_id", ""),
                    title=md.get("title", ""),
                    start_seconds=start,
                    timestamp=_timestamp(start),
                    deep_link=_deep_link(
                        md.get("url", ""), md.get("platform", "youtube"), start
                    ),
                    text=md.get("text", ""),
                    text_ts=md.get("text_ts") or md.get("text", ""),
                    # Pinecone gives back whatever was stored; a list is
                    # what this writes, but a malformed row must not take
                    # a search down, so anything else becomes empty.
                    speakers=[str(x) for x in (md.get("speakers") or [])
                              if isinstance(x, str)],
                    published_at=md.get("published_at"),
                    # Below every vector hit, so that if the reranker is off
                    # or fails these sit at the back rather than displacing
                    # a result the embedding actually chose.
                    score=0.0,
                )
            )
            added += 1
        if added:
            logger.info("exact-token match added %d passage(s) for %r",
                        added, query[:60])
        return hits

    async def _retrieve(self, query: str, top_k: int) -> list[PodcastHit]:
        # If the question names a host, search what that host actually
        # said. Without this the ranking is decided by topic alone, and
        # for "what did banks say about solana" the six best Solana
        # passages are all Ansem's — he has 361 Solana segments to Banks's
        # 117. The model then reports, correctly and uselessly, that it
        # cannot find Banks discussing Solana. It was reading the wrong
        # six passages.
        #
        # A metadata filter, not a re-ranking trick: the embeddings are
        # untouched and the same vector search runs, over the subset of
        # passages where that person speaks.
        speaker = host_named_in(query)
        # Cached, retry-on-rate-limit query embedding — repeat queries are
        # free and a rate-limited one backs off instead of hard-failing.
        vector = await embed_query(
            self._voyage,
            query,
            model=self._settings.voyage_model,
            dimension=self._settings.embedding_dimension,
        )

        # Pull a wider candidate set when reranking is on; the reranker
        # narrows it back down to top_k by actual relevance.
        fetch_k = (
            max(self._settings.rerank_candidates, top_k)
            if self._settings.rerank_model
            else top_k
        )

        def _query(restrict: str | None):
            kwargs = {
                "vector": vector,
                "top_k": fetch_k,
                "namespace": self._namespace,
                "include_metadata": True,
            }
            if restrict:
                kwargs["filter"] = {"speakers": {"$in": [restrict]}}
            return self.index.query(**kwargs)

        # Bounded like the upsert above, and for the same half-open-socket
        # reason. A read is the more dangerous case: it is on the request path
        # and holds a thread from the bounded to_thread pool while it hangs.
        response = await asyncio.wait_for(
            asyncio.to_thread(_query, speaker),
            timeout=self._settings.pinecone_read_timeout_seconds,
        )
        # Ask again unfiltered when the filter found nothing worth having.
        # Only half the archive carries speaker labels, so a question about
        # a host whose passages are all unlabelled would otherwise return
        # nothing at all — worse than the topic-ranked answer it replaces.
        if speaker and len(getattr(response, "matches", []) or []) < 3:
            logger.info("speaker filter for %r returned too little — "
                        "falling back to the whole archive", speaker)
            response = await asyncio.wait_for(
                asyncio.to_thread(_query, None),
                timeout=self._settings.pinecone_read_timeout_seconds,
            )
        hits: list[PodcastHit] = []
        for match in response.matches:
            if match.score < self._settings.retrieval_min_score:
                continue
            md = match.metadata or {}
            start = float(md.get("start_seconds", 0))
            hits.append(
                PodcastHit(
                    episode_id=md.get("episode_id", ""),
                    title=md.get("title", ""),
                    start_seconds=start,
                    timestamp=_timestamp(start),
                    deep_link=_deep_link(
                        md.get("url", ""), md.get("platform", "youtube"), start
                    ),
                    text=md.get("text", ""),
                    # Only the model reads this. It falls back to the plain
                    # text so vectors written before this existed still
                    # answer correctly, just with the old coarse citation.
                    text_ts=md.get("text_ts") or md.get("text", ""),
                    # Pinecone gives back whatever was stored; a list is
                    # what this writes, but a malformed row must not take
                    # a search down, so anything else becomes empty.
                    speakers=[str(x) for x in (md.get("speakers") or [])
                              if isinstance(x, str)],
                    published_at=md.get("published_at"),
                    score=match.score,
                )
            )

        # Exact-token candidates, added to the pool the reranker scores.
        #
        # Strictly additive by design. This never reorders, never drops, and
        # never overrides the vector search — it can only put one more
        # passage in front of the reranker, which is far better at judging
        # relevance than any keyword rule. A rare name or number is one word
        # in four hundred and barely moves an embedding, so the passage that
        # literally contains it can rank below passages merely about the same
        # subject: "who made 54 million on the drop" missed a line reading
        # "54 million dollars on the drop".
        hits = await self._add_exact_matches(query, hits)

        # Rerank by actual relevance (falls back to vector order on failure).
        keep = top_k
        if self._settings.rerank_model and len(hits) > top_k:
            # Title first, same as at ingest. Reranking the transcript
            # alone throws away the title signal the embedding just
            # used, so an episode found *because* of its title gets
            # demoted by the stage meant to improve the ordering.
            candidates = hits
            docs = [f"{h.title}\n\n{h.text}" for h in candidates]
            order = await rerank_order(
                self._voyage, query, docs,
                top_k=top_k, model=self._settings.rerank_model,
            )
            if order is not None:
                hits = [candidates[i] for i in order]

                # A deep candidate set reaches passages a shallow one
                # cannot -- the line naming who sold their entire ETH
                # position sits at rank 44 -- but reranking the deep set
                # alone loses answers the shallow one got right, because
                # positions two to six fill with passages merely about
                # the same subject and evict the specific one.
                #
                # Sequencing the two does nothing: reranking is a total
                # order, so narrowing fifty to twelve and reranking those
                # twelve gives back the same six. Measured, not assumed.
                # Combining them is what changes anything, because then
                # neither set has to win the same slots.
                narrow = self._settings.rerank_narrow_pool
                if narrow and len(candidates) > narrow:
                    shallow = await rerank_order(
                        self._voyage, query, docs[:narrow],
                        top_k=top_k, model=self._settings.rerank_model,
                    )
                    if shallow is not None:
                        # Indices address `candidates`, which is the order
                        # before the deep rerank rewrote `hits`.
                        def key(h: PodcastHit) -> tuple[str, float]:
                            return (h.episode_id, h.start_seconds)

                        deep = hits[:top_k]
                        seen = {key(h) for h in deep}
                        extra = [candidates[i] for i in shallow
                                 if key(candidates[i]) not in seen]
                        hits = deep + extra
                        # The union is pointless if the caller's slice
                        # throws it away again.
                        keep = len(deep) + len(extra)
        return _prefer_seekable(hits)[:keep]

    @staticmethod
    def _format(hits: list[PodcastHit]) -> str:
        if not hits:
            return "<excerpts>\n(nothing indexed matched this query)\n</excerpts>"
        # Chronological, so a topic reads in the order it was discussed.
        # Relevance order is what the hit list shows the user; the model
        # needs the timeline. Undated excerpts sort last rather than
        # inventing a position for them.
        hits = sorted(hits, key=lambda h: (h.published_at is None,
                                           h.published_at or "", h.start_seconds))
        # Transcript text/titles are untrusted third-party captions. XML-escape
        # them so a crafted window can't forge a closing </excerpt> tag and
        # break out of the data region the system prompt treats as grounding.
        blocks = [
            f"<excerpt episode={quoteattr(h.title)} at={quoteattr(h.timestamp)}"
            + (f" aired={quoteattr(h.published_at)}" if h.published_at else "")
            # Which hosts were detected speaking in this passage. Held in
            # the index since the labelling run, returned to the browser,
            # and until now never shown to the model -- which was being
            # asked by rules 5b-5d to attribute from name prefixes that
            # appear in none of the 91,190 lines. With nothing to attribute
            # from, every host-named question could only resolve one way:
            # "banks on polymarket" refused on eleven good hits, one of
            # them Banks explaining Polymarket for four minutes.
            + (f" voices={quoteattr(', '.join(h.speakers))}"
               if h.speakers else "")
            # Prefer the per-line timestamped copy so the model can cite the
            # line it used. Falls back to the plain text for anything
            # indexed before that field existed.
            + f">\n{escape(h.text_ts or h.text)}\n</excerpt>"
            for h in hits
        ]
        return "<excerpts>\n" + "\n\n".join(blocks) + "\n</excerpts>"

    REFUSAL_ANSWER = REFUSAL_ANSWER  # class alias for callers

    def _build_request(self, query: str, hits: list[PodcastHit],
                       instruction: str | None = None) -> dict:
        # A search answer is short and grounded — keep the request minimal
        # and fast. Config knobs are model-specific, so add them per family:
        #  - Haiku 4.5: no `effort` (400s) and no thinking → cheapest/fastest
        #  - Sonnet 5 / Opus 4.6+: effort + thinking disabled
        #  - Fable 5: thinking always on (omit), plus refusal fallback
        model = self._settings.search_model
        request: dict = {
            "model": model,
            "max_tokens": self._settings.search_max_tokens,
            "system": [
                {"type": "text",
                 "text": getattr(self, "_system_prompt", SYSTEM_PROMPT),
                 "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._format(hits)},
                        {"type": "text", "text": query,
                         "cache_control": {"type": "ephemeral"}},
                        # Per-surface style, added to the user turn rather
                        # than the system prompt so SYSTEM_PROMPT's bytes —
                        # and therefore its cache entry — stay identical
                        # across every caller.
                        *([{"type": "text", "text": instruction}]
                          if instruction else []),
                    ],
                }
            ],
        }
        if model.startswith("claude-fable"):
            request["betas"] = ["server-side-fallback-2026-06-01"]
            request["fallbacks"] = [
                {"model": self._settings.anthropic_fallback_model}
            ]
        elif "haiku" not in model:
            # Opus 4.6+ / Sonnet 5 support effort + adaptive thinking; a
            # grounded summary needs no reasoning, so thinking off + low.
            request["output_config"] = {"effort": self._settings.search_effort}
            request["thinking"] = {"type": "disabled"}
        return request

    def _record(self, model: str, usage) -> None:
        """Book one model call. Accounting must never break a search."""
        if self._ledger is None or usage is None:
            return
        try:
            self._ledger.record(self.SURFACE, model, {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read_input_tokens":
                    getattr(usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens":
                    getattr(usage, "cache_creation_input_tokens", 0),
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("usage accounting failed: %s", exc)

    async def retrieve(self, query: str, top_k: int | None = None) -> list[PodcastHit]:
        return await self._retrieve(query, top_k or self._settings.retrieval_top_k)

    async def search(
        self, query: str, top_k: int | None = None,
        instruction: str | None = None,
    ) -> PodcastSearchResponse:
        """Answer `query` from the index.

        `instruction` adds a per-surface style note for the model only. It is
        deliberately not part of `query`: the query is what gets embedded,
        and appending prose to it dilutes the vector and changes what comes
        back — measured, on this corpus, as the difference between finding a
        guest and missing them.
        """
        hits = await self.retrieve(query, top_k)
        primary, can_fall_back = self._llm()
        request = self._build_request(query, hits, instruction)
        try:
            response = await primary.with_options(
                timeout=self._settings.search_timeout_seconds
            ).beta.messages.create(**request)
        except Exception as exc:                              # noqa: BLE001
            if not can_fall_back:
                raise
            self._proxy_broke(exc)
            response = await self._fallback.with_options(
                timeout=self._settings.search_timeout_seconds
            ).beta.messages.create(**request)
        self._record(response.model, response.usage)
        if response.stop_reason == "refusal":
            return PodcastSearchResponse(
                answer=self.REFUSAL_ANSWER, hits=[],
                model=response.model, refused=True,
            )
        answer = "".join(
            b.text for b in response.content if b.type == "text"
        )
        # Same repair the stream above does, so the two endpoints cannot
        # disagree about what the answer to a question is. A real refusal
        # cites nothing and is left whole.
        answer, removed = hedging.strip_denial(answer)
        if removed:
            logger.info("dropped a denial the answer contradicts: %r",
                        removed[:80])
        return PodcastSearchResponse(answer=answer, hits=hits, model=response.model)

    # How much of the opening to hold before deciding. Every denial this has
    # ever produced announces itself in the first fifteen characters -- "I
    # couldn't find", "I don't see", "There is no specific" -- so the whole
    # first sentence is far more than is needed to tell them apart, and
    # waiting for one delayed every answer that was never broken. Sixty-four
    # characters is about a fifth of a second of tokens.
    _DENIAL_PEEK_CHARS = 64

    async def answer_stream(self, query: str, hits: list[PodcastHit]):
        """Yield answer text deltas for already-retrieved hits (SSE path).

        An answer that opens by denying what it then goes on to say is held
        back and repaired before any of it is shown. The X bot has done this
        since hedging.py existed, but it ran only there, so the website —
        the surface people are actually sent to — still opened with "I
        couldn't find that in the episodes I've indexed" and then answered
        the question underneath it. A reader takes the first line and
        scrolls.

        It cannot be fixed after the fact here the way it is on X, because
        the denial has already been streamed by the time the contradicting
        citation arrives. So the opening is examined first, and only an
        answer that starts with a denial waits for the rest before anything
        is sent. Every other answer streams exactly as it did.
        """
        primary, can_fall_back = self._llm()
        request = self._build_request(query, hits)

        # Opening the stream is where a broken proxy shows itself — a bad
        # token, an empty balance, nothing listening. Falling back here is
        # safe because not a byte has reached the reader yet. Once text is
        # flowing the offer is withdrawn: restarting mid-answer would
        # either repeat what was already on screen or splice two different
        # answers together, and a visible failure beats a quiet lie.
        try:
            opener = primary.with_options(
                timeout=self._settings.search_timeout_seconds
            ).beta.messages.stream(**request)
            entered = await opener.__aenter__()
        except Exception as exc:                              # noqa: BLE001
            if not can_fall_back:
                raise
            self._proxy_broke(exc)
            opener = self._fallback.with_options(
                timeout=self._settings.search_timeout_seconds
            ).beta.messages.stream(**request)
            entered = await opener.__aenter__()

        opening = ""
        decided = False        # have we judged the opening yet?
        holding = False        # opened with a denial, so buffer it all
        buffer = ""

        async with contextlib.AsyncExitStack() as guard:
            guard.push_async_exit(opener)
            stream = entered
            async for text in stream.text_stream:
                if not decided:
                    opening += text
                    # Whichever comes first. A short answer that ends before
                    # sixty-four characters still gets judged, by the branch
                    # after the loop.
                    if not (len(opening) >= self._DENIAL_PEEK_CHARS
                            or _SENTENCE_BREAK.search(opening)):
                        continue
                    decided = True
                    if hedging.opens_with_denial(opening):
                        holding, buffer = True, opening
                    else:
                        yield opening
                    continue
                if holding:
                    buffer += text
                else:
                    yield text

            if not decided:
                # The whole answer was shorter than one sentence.
                decided, buffer = True, opening
                holding = hedging.opens_with_denial(opening)
                if not holding:
                    yield opening

            if holding:
                repaired, removed = hedging.strip_denial(buffer)
                if removed:
                    logger.info("dropped a denial the answer contradicts: %r",
                                removed[:80])
                yield repaired

            final = await stream.get_final_message()
        self._record(final.model, final.usage)
        if final.stop_reason == "refusal":
            yield "\x00REFUSAL\x00"
