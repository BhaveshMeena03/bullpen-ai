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


def _annotated_transcript(episode: Episode, max_chars: int = 350_000) -> str:
    """Transcript with [h:mm:ss] markers so the model can cite timestamps."""
    lines = [f"[{_fmt_ts(seg.t)}] {seg.text}" for seg in episode.segments]
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
    async def summarize(self, episode: Episode) -> str:
        response = await self._anthropic.messages.create(
            model=self._settings.summary_model,
            max_tokens=2000,
            system=SUMMARY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Episode: {episode.title}\n\n"
                        f"{_annotated_transcript(episode)}"
                    ),
                }
            ],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused to summarize {episode.episode_id}")
        return "".join(b.text for b in response.content if b.type == "text")

    async def store(self, episode: Episode, summary: str) -> None:
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

        await asyncio.to_thread(_upsert)

    # -- serving --------------------------------------------------------------
    async def exists(self, episode_id: str) -> bool:
        def _fetch():
            return self.index.fetch(
                ids=[f"summary-{episode_id}"], namespace=NAMESPACE
            )
        result = await asyncio.to_thread(_fetch)
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

        rows = await asyncio.to_thread(_load)
        rows.sort(key=lambda r: r.get("published_at", ""), reverse=True)
        return rows
