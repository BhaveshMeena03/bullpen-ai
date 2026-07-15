"""API layer — asynchronous FastAPI server for the Bullpen Concierge.

Endpoints:
    POST /v1/chat         -> full JSON response with sources + usage
    POST /v1/chat/stream  -> Server-Sent Events token stream
    POST /v1/ingest       -> (internal) push documents into the vector DB
    GET  /healthz         -> liveness probe
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from voyageai import error as voyage_error

from .agent import REFUSAL_MESSAGE, ConciergeAgent
from .ingest import IngestionPipeline
from .podcast import PodcastIndex
from .retriever import Retriever
from .schemas import (
    ChatRequest,
    ChatResponse,
    Episode,
    IngestDocument,
    PodcastSearchRequest,
    PodcastSearchResponse,
)
from .security import global_rate_limit, public_rate_limit, require_admin
from .summaries import SummaryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build heavyweight clients once, at startup, and share them.
    app.state.retriever = Retriever()
    app.state.agent = ConciergeAgent()
    app.state.pipeline = IngestionPipeline()
    app.state.podcast = PodcastIndex()
    app.state.summaries = SummaryStore()
    yield


app = FastAPI(
    title="Bullpen Concierge",
    version="1.0.0",
    lifespan=lifespan,
)

# The widget is embedded on a different origin (the host site), so the
# browser needs CORS approval to call this API. Lock allow_origins down to
# the real host domains (e.g. https://app.bullpen.fi) before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["content-type"],
)

# Serve the embeddable widget and the demo terminal page from this same
# process: /widget/bullpen-concierge.js and /demo/.
_ROOT = Path(__file__).resolve().parent.parent
app.mount("/widget", StaticFiles(directory=_ROOT / "widget"), name="widget")
app.mount("/demo", StaticFiles(directory=_ROOT / "demo", html=True), name="demo")


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_agent(request: Request) -> ConciergeAgent:
    return request.app.state.agent


def get_pipeline(request: Request) -> IngestionPipeline:
    return request.app.state.pipeline


def get_podcast(request: Request) -> PodcastIndex:
    return request.app.state.podcast


def get_summaries(request: Request) -> SummaryStore:
    return request.app.state.summaries


@app.exception_handler(voyage_error.RateLimitError)
async def _voyage_rate_limit(request: Request, exc: voyage_error.RateLimitError):
    # Embedding quota exhausted (free-tier 3 RPM, or a spike). Fail soft.
    logger.warning("Voyage rate limit on %s", request.url.path)
    return JSONResponse(
        status_code=503,
        content={"detail": "Search is busy right now — try again in a moment."},
        headers={"Retry-After": "10"},
    )


@app.exception_handler(voyage_error.VoyageError)
async def _voyage_error(request: Request, exc: voyage_error.VoyageError):
    logger.error("Voyage error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=502, content={"detail": "Embedding provider error."}
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    # Last-resort net: an unexpected error returns a clean JSON 500, never
    # a raw stack-trace page. Logged with the path for debugging.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500, content={"detail": "Something went wrong. Please retry."}
    )


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    # Bare domain -> the Market Bubble search page (the public entry point).
    return RedirectResponse(url="/demo/podcast.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse,
          dependencies=[Depends(public_rate_limit), Depends(global_rate_limit)])
async def chat(
    body: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    agent: ConciergeAgent = Depends(get_agent),
) -> ChatResponse:
    chunks = await retriever.search(body.message, filters=body.filters)
    try:
        return await agent.answer(body.message, body.history, chunks)
    except anthropic.RateLimitError as exc:
        raise HTTPException(
            status_code=429, detail="Upstream rate limit; retry shortly."
        ) from exc
    except anthropic.APIStatusError as exc:
        logger.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        raise HTTPException(status_code=502, detail="Model provider error.") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(
            status_code=503, detail="Model provider unreachable."
        ) from exc


@app.post("/v1/chat/stream", dependencies=[Depends(public_rate_limit), Depends(global_rate_limit)])
async def chat_stream(
    body: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    agent: ConciergeAgent = Depends(get_agent),
) -> StreamingResponse:
    chunks = await retriever.search(body.message, filters=body.filters)

    async def event_source():
        # Sources first so the UI can render citations immediately.
        sources = [
            {"id": c.id, "source_type": c.source_type.value, "score": c.score,
             "metadata": c.metadata}
            for c in chunks
        ]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        try:
            async for delta in agent.stream(body.message, body.history, chunks):
                if delta == "\x00REFUSAL\x00":
                    # Whole fallback chain refused mid-stream: the partial
                    # text is invalid — tell the client to replace it.
                    payload = json.dumps({"text": REFUSAL_MESSAGE})
                    yield f"event: refusal\ndata: {payload}\n\n"
                    return
                yield f"data: {json.dumps({'text': delta})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except anthropic.APIError as exc:
            logger.error("Streaming failure: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': 'stream failed'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/ingest", dependencies=[Depends(require_admin)])
async def ingest(
    docs: list[IngestDocument],
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> dict:
    """Admin endpoint — requires X-Admin-Token when ADMIN_TOKEN is set."""
    count = await pipeline.ingest(docs)
    return {"chunks_upserted": count}


@app.post("/v1/podcast/search", response_model=PodcastSearchResponse,
          dependencies=[Depends(public_rate_limit), Depends(global_rate_limit)])
async def podcast_search(
    body: PodcastSearchRequest,
    podcast: PodcastIndex = Depends(get_podcast),
) -> PodcastSearchResponse:
    try:
        return await podcast.search(body.query, top_k=body.top_k)
    except anthropic.RateLimitError as exc:
        raise HTTPException(
            status_code=429, detail="Rate limited; retry shortly."
        ) from exc
    except anthropic.APIError as exc:
        # TEMP debug: surface the Anthropic error class + status so a live
        # 502 is diagnosable (no message body → no secret leak).
        status = getattr(exc, "status_code", None)
        logger.error("Anthropic error on search: %s (%s)", type(exc).__name__, status)
        raise HTTPException(
            status_code=502,
            detail=f"Model provider error [{type(exc).__name__}:{status}]",
        ) from exc


@app.post("/v1/podcast/ingest", dependencies=[Depends(require_admin)])
async def podcast_ingest(
    episodes: list[Episode],
    podcast: PodcastIndex = Depends(get_podcast),
) -> dict:
    """Admin endpoint — requires X-Admin-Token when ADMIN_TOKEN is set."""
    count = await podcast.ingest(episodes)
    return {"windows_indexed": count}


@app.get("/v1/podcast/episodes", dependencies=[Depends(public_rate_limit)])
async def podcast_episodes(
    summaries: SummaryStore = Depends(get_summaries),
) -> list[dict]:
    """Pre-computed episode summaries — a Pinecone fetch, no model call."""
    return await summaries.list_all()
