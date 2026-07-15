"""Ingestion & embedding pipeline.

Chunks three flavors of unstructured content — Markdown docs, podcast
transcripts, and scraped tweets — tags each chunk with metadata, embeds
via Voyage AI, and upserts into Pinecone.

Each source type gets its own chunking strategy:

- Markdown docs   -> split on headings, then size-bound with overlap.
- Transcripts     -> split on speaker turns, packed into windows.
- Tweets          -> one chunk per tweet (they are already atomic).
"""

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field

import voyageai
from pinecone import Pinecone

from .config import get_settings
from .embeddings import embed_texts
from .schemas import IngestDocument, SourceType

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_SPEAKER_RE = re.compile(r"^([A-Z][\w .'-]{1,40}):\s", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------

def _pack(sections: list[str], max_chars: int, overlap: int) -> list[str]:
    """Greedily pack sections into windows of at most max_chars,
    carrying a small overlap between consecutive windows."""
    windows: list[str] = []
    current = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(current) + len(section) + 1 <= max_chars:
            current = f"{current}\n{section}".strip()
        else:
            if current:
                windows.append(current)
            tail = current[-overlap:] if current else ""
            # A single oversized section is hard-split.
            while len(section) > max_chars:
                windows.append(f"{tail}\n{section[:max_chars]}".strip())
                tail = section[max_chars - overlap:max_chars]
                section = section[max_chars:]
            current = f"{tail}\n{section}".strip()
    if current:
        windows.append(current)
    return windows


def chunk_markdown(doc: IngestDocument, max_chars: int, overlap: int) -> list[Chunk]:
    """Split on headings so each chunk stays topically coherent."""
    boundaries = [m.start() for m in _HEADING_RE.finditer(doc.text)]
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    sections = [
        doc.text[start:end].strip()
        for start, end in zip(
            boundaries, boundaries[1:] + [len(doc.text)], strict=True
        )
    ]
    return [
        Chunk(text=window, metadata={"strategy": "markdown_heading"})
        for window in _pack(sections, max_chars, overlap)
    ]


def chunk_transcript(doc: IngestDocument, max_chars: int, overlap: int) -> list[Chunk]:
    """Split on speaker turns ("Ansem: ...", "Banks: ...") so quotes are
    never severed mid-utterance, then pack turns into windows."""
    positions = [m.start() for m in _SPEAKER_RE.finditer(doc.text)]
    if positions:
        if positions[0] != 0:
            positions.insert(0, 0)
        turns = [
            doc.text[start:end].strip()
            for start, end in zip(
                positions, positions[1:] + [len(doc.text)], strict=True
            )
        ]
    else:  # No speaker labels — fall back to paragraph packing.
        turns = doc.text.split("\n\n")
    return [
        Chunk(text=window, metadata={"strategy": "transcript_turns"})
        for window in _pack(turns, max_chars, overlap)
    ]


def chunk_tweets(doc: IngestDocument, max_chars: int, overlap: int) -> list[Chunk]:
    """Tweets are atomic. One tweet (or one thread item) per chunk;
    `doc.text` may contain several tweets separated by blank lines."""
    tweets = [t.strip() for t in doc.text.split("\n\n") if t.strip()]
    return [
        Chunk(text=tweet[:max_chars], metadata={"strategy": "tweet_atomic"})
        for tweet in tweets
    ]


_CHUNKERS = {
    SourceType.DOCS: chunk_markdown,
    SourceType.PODCAST: chunk_transcript,
    SourceType.TWEET: chunk_tweets,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """Chunk -> embed -> upsert. Embedding runs async (Voyage AsyncClient);
    Pinecone upserts run in a worker thread to keep the event loop free."""

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

    def _chunk(self, doc: IngestDocument) -> list[Chunk]:
        chunker = _CHUNKERS[doc.source_type]
        chunks = chunker(
            doc,
            self._settings.chunk_max_chars,
            self._settings.chunk_overlap_chars,
        )
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "source_type": doc.source_type.value,
                    "source_id": doc.source_id,
                    "chunk_index": i,
                    "text": chunk.text,
                }
            )
            # Optional fields only when present — Pinecone rejects nulls.
            for key in ("title", "author", "url", "published_at"):
                value = getattr(doc, key)
                if value:
                    chunk.metadata[key] = value
        return chunks

    async def ingest(self, docs: list[IngestDocument]) -> int:
        """Ingest a batch of documents. Returns the number of chunks upserted."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self._chunk(doc))
        if not all_chunks:
            return 0

        embeddings = await embed_texts(
            self._voyage,
            [c.text for c in all_chunks],
            model=self._settings.voyage_model,
            dimension=self._settings.embedding_dimension,
            input_type="document",
        )

        vectors = [
            {
                # Deterministic id: re-ingesting the same content overwrites
                # rather than duplicating.
                "id": hashlib.sha256(
                    f"{c.metadata['source_id']}:{c.metadata['chunk_index']}"
                    f":{c.text}".encode()
                ).hexdigest()[:32],
                "values": embedding,
                "metadata": c.metadata,
            }
            for c, embedding in zip(all_chunks, embeddings, strict=True)
        ]

        # Pinecone's client is synchronous; batch and offload to a thread.
        def _upsert() -> None:
            for start in range(0, len(vectors), 100):
                self.index.upsert(vectors=vectors[start:start + 100])

        await asyncio.to_thread(_upsert)
        logger.info("Upserted %d chunks from %d documents", len(vectors), len(docs))
        return len(vectors)
