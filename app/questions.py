"""What people actually ask, including the things this cannot answer.

The answered questions are the cheap half. The misses are the useful
half: every question the archive could not answer is either an episode
worth indexing, a name spelled a way retrieval does not recognise, or a
thing people expect this to do and it does not. None of that is visible
from the code, and none of it is guessable.

It lives in Pinecone rather than on disk because the service's disk is
ephemeral — every deploy clears it, which is fine for a resume marker
and useless for something meant to accumulate over months. Same
namespace-with-a-placeholder-vector shape as the summaries: these are
stored and listed, never similarity-searched.

Nothing here may break a reply. Every method swallows its own failures,
because a log that takes the product down with it when Pinecone has a
bad minute is worse than no log.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime

from pinecone import Pinecone

from app.config import get_settings

logger = logging.getLogger(__name__)

NAMESPACE = "questions"


class QuestionLog:
    def __init__(self) -> None:
        self._settings = get_settings()
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

    async def record(self, question: str, *, source: str = "x",
                     asker: str | None = None, answered: bool = True,
                     episode: str | None = None,
                     timestamp: str | None = None,
                     reference: str | None = None) -> None:
        """Note one question and whether the archive could answer it.

        `reference` is the mention id where there is one, so the same
        mention retried after a restart overwrites its own row rather
        than counting twice.
        """
        question = (question or "").strip()
        if not question:
            return

        when = datetime.now(UTC).isoformat()
        key = reference or hashlib.sha256(
            f"{source}:{asker}:{question}:{when[:13]}".encode()).hexdigest()[:24]

        record = {
            "question": question[:500],
            "source": source,
            "answered": answered,
            "asked_at": when,
        }
        if asker:
            record["asker"] = asker
        if episode:
            record["episode"] = episode
        if timestamp:
            record["timestamp"] = timestamp

        def _upsert() -> None:
            self.index.upsert(
                vectors=[{"id": f"q-{key}",
                          "values": self._placeholder_vector(),
                          "metadata": record}],
                namespace=NAMESPACE,
            )

        try:
            await asyncio.to_thread(_upsert)
        except Exception as exc:                                # noqa: BLE001
            # Deliberately swallowed: see the module docstring. A reply
            # already composed must not be lost to a logging failure.
            logger.warning("could not record the question (%s)", exc)

    async def list_all(self, limit: int = 2000) -> list[dict]:
        """Every question recorded, newest first."""
        def _read() -> list[dict]:
            # index.list() yields ListResponse pages and the ids live on
            # .vectors — iterating the page directly reads back nothing at
            # all, silently, which is how the first version of this
            # "worked" while logging every question into a hole.
            ids = [
                item.id if hasattr(item, "id") else str(item)
                for page in self.index.list(namespace=NAMESPACE)
                for item in (page.vectors if hasattr(page, "vectors") else page)
            ][:limit]
            if not ids:
                return []
            out = []
            # Chunked: Pinecone caps ids per fetch, and a log that breaks
            # once it gets popular is not much of a log.
            for start in range(0, len(ids), 100):
                fetched = self.index.fetch(ids=ids[start:start + 100],
                                           namespace=NAMESPACE)
                for vector in (fetched.vectors or {}).values():
                    out.append(dict(vector.metadata or {}))
            return out

        try:
            # Bounded: the Pinecone client has no read timeout of its own,
            # and this holds a thread from the shared to_thread pool.
            rows = await asyncio.wait_for(
                asyncio.to_thread(_read),
                timeout=self._settings.pinecone_read_timeout_seconds,
            )
        except Exception as exc:                                # noqa: BLE001
            logger.warning("could not read the question log (%s)", exc)
            return []
        return sorted(rows, key=lambda r: r.get("asked_at", ""), reverse=True)
