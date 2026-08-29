"""Episode summaries — pre-computed once, served instantly.

A 3-4 hour episode transcript fits in a single Claude call, so each
episode is summarized ONE time (by scripts/summarize_episodes.py) and the
result is stored in a dedicated Pinecone namespace. Serving a summary is
then a fetch, not a model call — instant and free.

Summaries live in their own namespace with a fixed unit vector: they are
looked up by ID, never by similarity, and must not pollute search results.
"""

import asyncio
import logging

from anthropic import AsyncAnthropic
from pinecone import Pinecone

from .config import get_settings
from .schemas import Episode

logger = logging.getLogger(__name__)

NAMESPACE = "summaries"
MAX_METADATA_CHARS = 30_000  # Pinecone metadata cap is 40KB; stay well under

SUMMARY_SYSTEM = """\
You summarize episodes of the "Market Bubble" podcast (hosted by Ansem and \
FaZe Banks) from their transcripts. The transcript lines are prefixed with \
[h:mm:ss] timestamps.

Some lines also carry the speaker's name before the text, like "[12:02] \
FaZe Banks: I put close to seven figures in Hyperliquid". Where a name is \
there, use it — "Ansem argued X and Banks pushed back" is worth far more \
than "the hosts discussed X". Two rules about those names, and they are \
the same two the answering side follows because breaking either one puts \
words in a real person's mouth:

The prefix is the ONLY thing that establishes who spoke. Attribute a line \
to the name in front of it and to nobody else.

A name INSIDE a line is somebody being talked about, not the person \
talking. "FaZe Banks: I'm gonna help continue to guide Z" is Banks \
speaking about Ansem, not Ansem speaking.

Lines with no prefix are usually guests, who are not labelled at all. \
Describe them the way the episode does — by name if the title or an \
introduction makes it plain, otherwise as "a guest". Never borrow a \
host's name for an unprefixed line.

Produce a summary with exactly these sections, in Markdown:

**TL;DR** — 2-3 sentences: what this episode is about and the single most \
interesting thread.

**Topics** — 5-10 bullet points in chronological order. Each bullet starts \
with the approximate timestamp where the topic begins (taken from the \
transcript markers, format [h:mm:ss]) followed by a one-line description.

**Notable moments** — 2-4 bullets for the most quotable or surprising \
exchanges, each with its timestamp.

Rules:
1. Only state what the transcript supports — never invent quotes or \
attribute specific wording to a host; paraphrase.
2. These are auto-generated captions without speaker labels, so refer to \
"the hosts" or "a guest" unless identity is unambiguous from context.
3. This is an informational summary, not financial advice. Report opinions \
as opinions ("the hosts argue that...") and never add recommendations.
4. Keep the whole summary under 500 words."""


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _annotated_transcript(episode: Episode, max_chars: int = 350_000,
                          speakers: dict[str, str] | None = None) -> str:
    """Transcript with [h:mm:ss] markers so the model can cite timestamps.

    And a name in front of the line where one is known. Without it a
    summary can only say "the hosts discussed", because the transcript it
    reads has no idea who was talking — which is how twelve of the first
    thirty-three summaries came out with no names in them at all.

    Numbered the way the fingerprinting numbered them: non-empty text
    only, in order. A line with no entry keeps exactly the shape it had.
    """
    speakers = speakers or {}
    lines, numbered = [], 0
    for seg in episode.segments:
        stamp = _fmt_ts(seg.t)
        if not (seg.text or "").strip():
            lines.append(f"[{stamp}] {seg.text}")
            continue
        who = speakers.get(str(numbered))
        numbered += 1
        lines.append(f"[{stamp}] {who}: {seg.text}" if who
                     else f"[{stamp}] {seg.text}")
    text = "\n".join(lines)
    if len(text) > max_chars:  # ~100K tokens; extremely long episodes
        text = text[:max_chars] + "\n[transcript truncated]"
    return text


class SummaryStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._index = None

    @property
    def index(self):
        if self._index is None:
            self._index = Pinecone(
                api_key=self._settings.pinecone_api_key
            ).Index(self._settings.pinecone_index)
        return self._index

    def _placeholder_vector(self) -> list[float]:
        # Unit vector: valid for cosine metric, never similarity-searched.
        vec = [0.0] * self._settings.embedding_dimension
        vec[0] = 1.0
        return vec

    # -- generation (run once per episode, by the admin script) -------------
    async def summarize(self, episode: Episode,
                        speakers: dict[str, str] | None = None) -> str:
        response = await self._anthropic.messages.create(
            model=self._settings.summary_model,
            max_tokens=2000,
            system=SUMMARY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Episode: {episode.title}\n\n"
                        f"{_annotated_transcript(episode, speakers=speakers)}"
                    ),
                }
            ],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused to summarize {episode.episode_id}")
        summary = "".join(b.text for b in response.content if b.type == "text")
        # An empty summary stores cleanly and fails silently: the episode
        # looks summarised, and "summarize the latest episode" comes back as
        # a bare link with no text above it. That happened on the first
        # broadcast added after this pipeline existed.
        if len(summary.strip()) < 200:
            raise RuntimeError(
                f"summary for {episode.episode_id} came back with "
                f"{len(summary.strip())} characters — refusing to store it")
        return summary

    async def store(self, episode: Episode, summary: str) -> None:
        if len(summary.strip()) < 200:
            # Belt and braces: verify_summaries rewrites through this path
            # too, and dropping every topic line from a thin summary could
            # otherwise leave an empty one behind.
            raise ValueError(
                f"refusing to store a {len(summary.strip())}-character "
                f"summary for {episode.episode_id}")
        metadata = {
            "episode_id": episode.episode_id,
            "title": episode.title,
            "url": episode.url,
            "platform": episode.platform,
            "summary": summary[:MAX_METADATA_CHARS],
        }
        if episode.published_at:
            metadata["published_at"] = episode.published_at

        def _upsert() -> None:
            self.index.upsert(
                vectors=[{
                    "id": f"summary-{episode.episode_id}",
                    "values": self._placeholder_vector(),
                    "metadata": metadata,
                }],
                namespace=NAMESPACE,
            )

        await asyncio.wait_for(
            asyncio.to_thread(_upsert),
            timeout=self._settings.pinecone_write_timeout_seconds,
        )

    # -- serving --------------------------------------------------------------
    async def exists(self, episode_id: str) -> bool:
        def _fetch():
            return self.index.fetch(
                ids=[f"summary-{episode_id}"], namespace=NAMESPACE
            )
        # Bounded: the Pinecone client has no read timeout, and this sits
        # on the request path holding a thread from the bounded to_thread
        # pool. A hang here starves every offloaded call in the process.
        result = await asyncio.wait_for(
            asyncio.to_thread(_fetch),
            timeout=self._settings.pinecone_read_timeout_seconds,
        )
        return bool(result.vectors)

    async def list_all(self) -> list[dict]:
        """All episode summaries, newest first by published date."""
        def _load() -> list[dict]:
            # index.list() yields ListResponse pages; IDs live in .vectors.
            ids = [
                item.id if hasattr(item, "id") else str(item)
                for page in self.index.list(namespace=NAMESPACE)
                for item in (page.vectors if hasattr(page, "vectors") else page)
            ]
            if not ids:
                return []
            fetched = self.index.fetch(ids=ids, namespace=NAMESPACE)
            return [dict(v.metadata or {}) for v in fetched.vectors.values()]

        # Bounded: the Pinecone client has no read timeout, and this sits
        # on the request path holding a thread from the bounded to_thread
        # pool. A hang here starves every offloaded call in the process.
        rows = await asyncio.wait_for(
            asyncio.to_thread(_load),
            timeout=self._settings.pinecone_read_timeout_seconds,
        )
        rows.sort(key=lambda r: r.get("published_at", ""), reverse=True)
        return rows
