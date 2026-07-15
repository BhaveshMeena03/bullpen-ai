"""Pydantic models shared across the API, retriever, and agent layers."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """The three unstructured corpora the concierge is grounded in."""

    DOCS = "docs"          # Markdown product documentation
    PODCAST = "podcast"    # "Market Bubble" episode transcripts
    TWEET = "tweet"        # Scraped tweets (e.g. @blknoiz06, @bullpen)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=16000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    # Prior turns of the conversation; the API layer is stateless so the
    # client replays history. Keeping history byte-identical between turns
    # is what lets the prompt cache pay off. Bounded so a hostile client
    # can't stuff megabytes of fake history into every request.
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    # Optional metadata pre-filter, e.g. {"source_type": "docs"} to answer
    # only from documentation, or {"episode": "ep-42"}.
    filters: dict[str, str | int | bool] | None = None
    session_id: str | None = None


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    source_type: SourceType
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    model: str
    refused: bool = False
    usage: dict | None = None


class IngestDocument(BaseModel):
    """One raw document handed to the ingestion pipeline."""

    source_type: SourceType
    source_id: str                 # doc slug / episode id / tweet id
    text: str
    title: str | None = None
    author: str | None = None
    url: str | None = None
    published_at: str | None = None  # ISO 8601


# --- Market Bubble podcast search -----------------------------------------

class TranscriptSegment(BaseModel):
    """One timestamped line of a podcast transcript (as it comes out of a
    YouTube/Whisper caption file)."""

    t: float = Field(..., ge=0)   # start time in seconds
    text: str
    speaker: str | None = None


class Episode(BaseModel):
    """A Market Bubble episode with a timestamped transcript."""

    episode_id: str
    title: str
    url: str                       # base watch URL (YouTube/Spotify)
    platform: Literal["youtube", "spotify", "other"] = "youtube"
    published_at: str | None = None
    segments: list[TranscriptSegment]


class PodcastHit(BaseModel):
    """A retrieved transcript window, deep-linked to the moment in the episode."""

    episode_id: str
    title: str
    start_seconds: float
    timestamp: str                 # "14:32"
    deep_link: str                 # url that jumps to start_seconds
    text: str
    score: float


class PodcastSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = None


class PodcastSearchResponse(BaseModel):
    answer: str
    hits: list[PodcastHit]
    model: str
    refused: bool = False
