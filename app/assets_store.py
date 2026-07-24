"""Extracted asset mentions, stored per episode in Pinecone.

Same pattern as SummaryStore, and for the same reason: the deployed app
reads this at runtime, so a new episode's tokens appear on the dashboard
without a redeploy. A local JSON file would have gone stale the moment the
weekly sync ran on a different machine than the one serving traffic.

Hits are split across as many records as an episode needs. A single record
per episode was the first design and it silently truncated 8 of 15 episodes:
real ones serialize to 30-48KB, past Pinecone's 40KB metadata cap.
Aggregation across episodes happens at read time in app/assets.py.
"""

import asyncio
import json
import logging

from pinecone import Pinecone

from .config import get_settings

logger = logging.getLogger(__name__)

NAMESPACE = "assets"
MAX_METADATA_CHARS = 30_000  # Pinecone caps metadata at 40KB; stay under it


def _chunk_hits(hits: list[dict], budget: int) -> list[list[dict]]:
    """Split hits into groups that each serialize under `budget` chars.

    Sized incrementally (per-hit length + separator) rather than by
    re-serializing the whole list each step, so this stays linear.
    """
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    size = 2  # the enclosing "[]"
    for h in hits:
        hs = len(json.dumps(h, separators=(",", ":"))) + 1
        if cur and size + hs > budget:
            chunks.append(cur)
            cur, size = [], 2
        cur.append(h)
        size += hs
    if cur:
        chunks.append(cur)
    return chunks


class AssetStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
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

    async def store(self, episode_id: str, title: str, hits: list[dict]) -> int:
        """Persist one episode's raw hits, split across as many records as it
        takes. Returns how many were stored.

        A single record per episode was the original design and it silently
        lost data: real episodes serialize to 30-48KB, well past Pinecone's
        40KB metadata cap, so 8 of 15 episodes were being truncated. Chunking
        stores everything instead of dropping the tail.
        """
        chunks = _chunk_hits(hits, MAX_METADATA_CHARS)
        new_ids = {f"assets-{episode_id}-{i}" for i in range(len(chunks))}

        def _write() -> None:
            if chunks:
                self.index.upsert(
                    vectors=[{
                        "id": f"assets-{episode_id}-{i}",
                        "values": self._placeholder_vector(),
                        "metadata": {
                            "episode_id": episode_id,
                            "title": title,
                            "chunk": i,
                            "hits": json.dumps(chunk, separators=(",", ":")),
                        },
                    } for i, chunk in enumerate(chunks)],
                    namespace=NAMESPACE,
                )
            # Re-running with fewer hits (or migrating off the old unchunked
            # "assets-<id>" record) must not leave orphans behind.
            stale = [
                i for i in self._ids_for_episode(episode_id) if i not in new_ids
            ]
            if stale:
                self.index.delete(ids=stale, namespace=NAMESPACE)
                logger.info("assets: removed %d stale record(s) for %s",
                            len(stale), episode_id)

        # Same no-read-timeout hazard as the podcast upsert: bound the write so
        # a dead socket fails fast instead of hanging. Safe to retry — the store
        # upserts by deterministic id and prunes stale records.
        await asyncio.wait_for(
            asyncio.to_thread(_write),
            timeout=self._settings.pinecone_write_timeout_seconds,
        )
        return len(hits)

    def _ids_for_episode(self, episode_id: str) -> list[str]:
        prefix = f"assets-{episode_id}"
        return [
            i for i in self._all_ids()
            if i == prefix or i.startswith(prefix + "-")
        ]

    def _all_ids(self) -> list[str]:
        return [
            item.id if hasattr(item, "id") else str(item)
            for page in self.index.list(namespace=NAMESPACE)
            for item in (page.vectors if hasattr(page, "vectors") else page)
        ]

    async def exists(self, episode_id: str) -> bool:
        # Must match chunked ids ("assets-<id>-0"), not just the legacy
        # single-record id, or a stored episode reads back as missing.
        return bool(await asyncio.to_thread(self._ids_for_episode, episode_id))

    async def all_hits(self) -> list[dict]:
        """Every stored hit across every episode, ready to aggregate."""
        def _load() -> list[dict]:
            ids = self._all_ids()
            if not ids:
                return []
            fetched = self.index.fetch(ids=ids, namespace=NAMESPACE)
            hits: list[dict] = []
            for v in fetched.vectors.values():
                raw = (v.metadata or {}).get("hits", "[]")
                try:
                    hits.extend(json.loads(raw))
                except (ValueError, TypeError):
                    logger.warning("assets: unparseable hits blob, skipping")
            return hits

        return await asyncio.to_thread(_load)

    async def episode_count(self) -> int:
        def _count() -> int:
            return sum(
                1
                for page in self.index.list(namespace=NAMESPACE)
                for _ in (page.vectors if hasattr(page, "vectors") else page)
            )
        return await asyncio.to_thread(_count)
