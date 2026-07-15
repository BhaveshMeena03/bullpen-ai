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

import voyageai
from anthropic import AsyncAnthropic
from pinecone import Pinecone

from .config import get_settings
from .embeddings import embed_texts
from .schemas import (
    Episode,
    PodcastHit,
    PodcastSearchResponse,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

NAMESPACE = "podcast"

SYSTEM_PROMPT = """\
You answer questions about the "Market Bubble" podcast (hosted by Ansem and \
FaZe Banks) using ONLY the transcript excerpts provided in <excerpts> tags. \
Each excerpt is tagged with its episode and timestamp.

Rules:
1. Answer strictly from the excerpts. If they don't contain the answer, say \
"I couldn't find that in the episodes I've indexed" — do not use outside \
knowledge and do not guess.
2. Cite the moment: mention which episode and roughly when \
("around 14:30 in <episode>"). The interface shows clickable timestamps \
alongside your answer, so refer to them naturally.
3. Summarize faithfully. Do not put words in the hosts' mouths or invent \
quotes — paraphrase what the excerpt actually says.
4. This is an informational search tool, not financial advice. Never add \
buy/sell recommendations or price predictions of your own.
5. Keep it tight and conversational — a couple of sentences plus the \
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
    an answer that straddles a boundary is still retrievable."""
    windows: list[tuple[float, str]] = []
    i = 0
    n = len(segments)
    while i < n:
        start_t = segments[i].t
        parts: list[str] = []
        length = 0
        j = i
        while j < n and length + len(segments[j].text) + 1 <= max_chars:
            speaker = f"{segments[j].speaker}: " if segments[j].speaker else ""
            line = f"{speaker}{segments[j].text}"
            parts.append(line)
            length += len(line) + 1
            j += 1
        if j == i:  # single segment longer than max_chars — take it whole
            parts.append(segments[i].text[:max_chars])
            j = i + 1
        windows.append((start_t, "\n".join(parts)))
        if j >= n:
            break
        i = max(j - overlap_segments, i + 1)
    return windows


class PodcastIndex:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._voyage = voyageai.AsyncClient(api_key=settings.voyage_api_key)
        self._anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._index = None

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
            for start_t, text in _windows(
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
                    }
                )
        if not rows:
            return 0

        embeddings = await embed_texts(
            self._voyage,
            [r["text"] for r in rows],
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

        await asyncio.to_thread(_upsert)
        logger.info("Indexed %d transcript windows from %d episodes",
                    len(vectors), len(episodes))
        return len(vectors)

    # -- search -------------------------------------------------------------
    async def _retrieve(self, query: str, top_k: int) -> list[PodcastHit]:
        vector = (
            await self._voyage.embed(
                texts=[query],
                model=self._settings.voyage_model,
                input_type="query",
                output_dimension=self._settings.embedding_dimension,
            )
        ).embeddings[0]

        def _query():
            return self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=NAMESPACE,
                include_metadata=True,
            )

        response = await asyncio.to_thread(_query)
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
                    score=match.score,
                )
            )
        return hits

    @staticmethod
    def _format(hits: list[PodcastHit]) -> str:
        if not hits:
            return "<excerpts>\n(nothing indexed matched this query)\n</excerpts>"
        blocks = [
            f'<excerpt episode="{h.title}" at="{h.timestamp}">\n{h.text}\n</excerpt>'
            for h in hits
        ]
        return "<excerpts>\n" + "\n\n".join(blocks) + "\n</excerpts>"

    async def search(
        self, query: str, top_k: int | None = None
    ) -> PodcastSearchResponse:
        top_k = top_k or self._settings.retrieval_top_k
        hits = await self._retrieve(query, top_k)

        model = self._settings.anthropic_model
        request: dict = {
            "model": model,
            "max_tokens": self._settings.max_tokens,
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
                    ],
                }
            ],
            "output_config": {"effort": self._settings.effort},
        }
        if model.startswith("claude-fable"):
            request["betas"] = ["server-side-fallback-2026-06-01"]
            request["fallbacks"] = [
                {"model": self._settings.anthropic_fallback_model}
            ]
        else:
            request["thinking"] = {"type": "adaptive"}

        response = await self._anthropic.beta.messages.create(**request)
        if response.stop_reason == "refusal":
            return PodcastSearchResponse(
                answer="I can't help with that one — try asking about "
                       "something discussed on the show.",
                hits=[], model=response.model, refused=True,
            )
        answer = "".join(
            b.text for b in response.content if b.type == "text"
        )
        return PodcastSearchResponse(answer=answer, hits=hits, model=response.model)
