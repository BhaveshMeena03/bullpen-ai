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
import hashlib
import logging
import re
from xml.sax.saxutils import escape, quoteattr

import voyageai
from anthropic import AsyncAnthropic
from pinecone import Pinecone

from .config import get_settings
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

NAMESPACE = "podcast"

REFUSAL_ANSWER = ("I can't help with that one — try asking about "
                  "something discussed on the show.")

# What the model is told to say when the excerpts do not contain the answer.
# Callers need to recognise a miss, and the only signal is the wording: the
# retriever always returns its top_k, so a full hit list means nothing about
# whether any of it was relevant. Kept as a constant rather than interpolated
# into the prompt below, because SYSTEM_PROMPT's exact bytes are the prompt
# cache key — a test asserts the two stay in step.
NOT_FOUND_ANSWER = "I couldn't find that in the episodes I've indexed"

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
the ONLY thing that establishes who spoke. Attribute a line to the name \
in front of it and to nobody else. A line with no prefix has no known \
speaker — describe it as "one of the hosts" or "a guest", never as the \
person the question asked about. Asked what Banks said about Solana, the \
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


_SEEKABLE_HOST = re.compile(r"^https?://(www\.)?(youtube\.com|youtu\.be|open\.spotify\.com)/")


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
    """Where the same moment appears twice, keep the copy you can jump to.

    Roughly half of every live broadcast is also in the YouTube upload of
    that episode, so a single query can retrieve the same passage from
    both. They are not equivalent: a YouTube citation carries ?t= and lands
    on the second being quoted, while an X one cannot, because X has no
    timestamp parameter for video. Returning the X copy when the YouTube
    one exists costs the reader the entire point of the citation.

    Order is otherwise untouched — this only drops a later duplicate, and
    only when a seekable hit already covers it. A non-seekable hit with no
    seekable twin stays, because half-covered is better than missing.
    """
    kept: list[PodcastHit] = []
    for hit in hits:
        if _can_seek(hit.deep_link):
            kept.append(hit)
            continue
        covered = any(_can_seek(k.deep_link) and _same_moment(k.text, hit.text)
                      for k in hits)
        if not covered:
            kept.append(hit)
    return kept


class PodcastIndex:
    SURFACE = "market-bubble-search"

    def __init__(self, ledger=None) -> None:
        self._ledger = ledger
        settings = get_settings()
        self._settings = settings
        self._voyage = voyageai.AsyncClient(api_key=settings.voyage_api_key)
        self._anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._index = None
        # Loaded once. Absent or unreadable means every lookup returns
        # nothing and search behaves exactly as it did before.
        self._terms = TermIndex()

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
                    vectors=vectors[start:start + 100], namespace=NAMESPACE
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
        ids = self._terms.lookup(query)
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
            return self.index.fetch(ids=wanted, namespace=NAMESPACE)

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
                "namespace": NAMESPACE,
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
                {"type": "text", "text": SYSTEM_PROMPT,
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
        client = self._anthropic.with_options(
            timeout=self._settings.search_timeout_seconds
        )
        response = await client.beta.messages.create(
            **self._build_request(query, hits, instruction)
        )
        self._record(response.model, response.usage)
        if response.stop_reason == "refusal":
            return PodcastSearchResponse(
                answer=self.REFUSAL_ANSWER, hits=[],
                model=response.model, refused=True,
            )
        answer = "".join(
            b.text for b in response.content if b.type == "text"
        )
        return PodcastSearchResponse(answer=answer, hits=hits, model=response.model)

    async def answer_stream(self, query: str, hits: list[PodcastHit]):
        """Yield answer text deltas for already-retrieved hits (SSE path)."""
        client = self._anthropic.with_options(
            timeout=self._settings.search_timeout_seconds
        )
        async with client.beta.messages.stream(
            **self._build_request(query, hits)
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        self._record(final.model, final.usage)
        if final.stop_reason == "refusal":
            yield "\x00REFUSAL\x00"
