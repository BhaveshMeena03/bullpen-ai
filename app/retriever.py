"""Retrieval core.

Embeds the user query with Voyage AI and runs a similarity search against
Pinecone, with optional metadata pre-filtering (e.g. restrict to docs only,
or to a specific podcast episode).
"""

import asyncio
import logging

import voyageai
from pinecone import Pinecone

from .config import get_settings
from .embeddings import embed_query
from .schemas import RetrievedChunk, SourceType

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._voyage = voyageai.AsyncClient(api_key=settings.voyage_api_key)
        self._index = None  # resolved lazily — .Index() does network I/O

    @property
    def index(self):
        if self._index is None:
            self._index = Pinecone(
                api_key=self._settings.pinecone_api_key
            ).Index(self._settings.pinecone_index)
        return self._index

    @staticmethod
    def _build_filter(filters: dict | None) -> dict | None:
        """Translate the API's flat filter dict into Pinecone's filter DSL.

        {"source_type": "docs"} -> {"source_type": {"$eq": "docs"}}
        """
        if not filters:
            return None
        return {key: {"$eq": value} for key, value in filters.items()}

    async def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self._settings.retrieval_top_k

        query_vector = await embed_query(
            self._voyage,
            query,
            model=self._settings.voyage_model,
            dimension=self._settings.embedding_dimension,
        )

        def _query():
            return self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=self._build_filter(filters),
                include_metadata=True,
            )

        response = await asyncio.to_thread(_query)

        chunks: list[RetrievedChunk] = []
        for match in response.matches:
            if match.score < self._settings.retrieval_min_score:
                continue
            metadata = dict(match.metadata or {})
            text = metadata.pop("text", "")
            chunks.append(
                RetrievedChunk(
                    id=match.id,
                    text=text,
                    score=match.score,
                    source_type=SourceType(metadata.get("source_type", "docs")),
                    metadata=metadata,
                )
            )
        logger.debug("Retrieved %d chunks for query %r", len(chunks), query[:80])
        return chunks
