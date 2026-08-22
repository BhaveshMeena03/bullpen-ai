"""API layer — asynchronous FastAPI server for the Bullpen Concierge.

Endpoints:
    POST /v1/chat         -> full JSON response with sources + usage
    POST /v1/chat/stream  -> Server-Sent Events token stream
    POST /v1/ingest       -> (internal) push documents into the vector DB
    GET  /healthz         -> liveness probe
"""

import asyncio
import json
import logging
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from voyageai import error as voyage_error

from . import market
from .agent import REFUSAL_MESSAGE, ConciergeAgent
from .assets import aggregate as aggregate_assets
from .assets_store import AssetStore
from .config import get_settings
from .ingest import IngestionPipeline
from .podcast import REFUSAL_ANSWER as PODCAST_REFUSAL
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
from .security import (
    RateLimiter,
    daily_budget,
    global_rate_limit,
    per_client_daily,
    public_rate_limit,
    require_admin,
)
from .summaries import SummaryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lightweight usage counters (in-memory: reset on restart/redeploy — good
# enough for "is anyone using this?"). Structured log lines below are the
# durable record; grep the host's logs for ANALYTICS.
STATS: dict = {
    "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    "podcast_searches": 0,
    "concierge_chats": 0,
    "episode_summary_views": 0,
    "asset_dashboard_views": 0,
    "asset_detail_views": 0,
    "unanswered_chats": 0,
    "refusals": 0,
}


def _track(kind: str, **fields) -> None:
    STATS[kind] = STATS.get(kind, 0) + 1
    logger.info("ANALYTICS %s", json.dumps({"event": kind, **fields}))


# Feedback loop: the questions the concierge could NOT answer from the
# knowledge base. This is the list of docs to write next — the single most
# useful signal for improving a support agent over time. Bounded ring buffer;
# the durable record is the "ANALYTICS kb_gap" log lines.
GAPS: deque = deque(maxlen=200)

# Heuristic markers for the agent's tier-3 fallback — when it answers but
# admits it can't ground an operational specific (see the system prompt's
# "say you don't have that information" instruction). Kept deliberately narrow
# to avoid flagging normal answers that happen to contain these words.
_UNKNOWN_MARKERS = (
    "don't have that information",
    "do not have that information",
    "don't have specific",
    "don't have information on",
    "couldn't find that",
    "could not find that",
    "i don't have details",
    "contact official bullpen support",
    "official bullpen support channels",
    "reach out to official",
)


def _looks_unanswered(answer: str) -> bool:
    a = answer.lower()
    return any(marker in a for marker in _UNKNOWN_MARKERS)


def _record_gap(query: str, reason: str) -> None:
    """reason: 'no_context' (retriever found nothing) or 'low_confidence'
    (answered but couldn't ground an operational specific)."""
    STATS["unanswered_chats"] += 1
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "query": query[:300],
        "reason": reason,
    }
    GAPS.append(record)
    logger.info("ANALYTICS %s", json.dumps({"event": "kb_gap", **record}))


def _classify_outcome(query: str, chunks: list, answer: str, refused: bool) -> None:
    """Record a KB gap when the concierge couldn't help. A safety *refusal*
    is the guardrail working, not a missing doc, so it's counted separately
    and never treated as a gap."""
    if refused:
        STATS["refusals"] += 1
        return
    if not chunks:
        _record_gap(query, "no_context")
    elif _looks_unanswered(answer):
        _record_gap(query, "low_confidence")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build heavyweight clients once, at startup, and share them.
    app.state.retriever = Retriever()
    app.state.agent = ConciergeAgent()
    app.state.pipeline = IngestionPipeline()
    app.state.podcast = PodcastIndex()
    app.state.summaries = SummaryStore()
    app.state.assets = AssetStore()
    # Bounded per-ticker cache for live market lookups, and the CoinGecko
    # symbol table. Both fill lazily — startup must not wait on a third party.
    app.state._market_cache = {}
    app.state._cg_table = None
    yield


app = FastAPI(
    title="Bullpen Concierge",
    version="1.0.0",
    lifespan=lifespan,
)

# The widget is embedded on a different origin (the host site), so the browser
# needs CORS approval to call this API.
#
# "*" is the deliberate choice here, not an unfinished TODO. What makes a
# wildcard dangerous is pairing it with credentials: the browser then attaches
# the visitor's cookies to a cross-origin call and any site can act as them.
# This API has no cookie or session auth at all, so allow_credentials stays
# off (the default) and a cross-origin request can do nothing a plain curl
# could not already do. Verified in production: the preflight returns
# access-control-allow-origin with no allow-credentials header.
#
# Two things must stay true for that reasoning to hold, so change them only
# together with this comment:
#   - never set allow_credentials=True while origins is "*"
#   - keep x-admin-token OUT of allow_headers, so a page in someone's browser
#     cannot be made to carry an admin token to the ingest endpoints
#
# The remaining cost of an open origin is that any site could embed the widget
# and spend model budget. That is bounded by the global RPM ceiling and the
# daily request budget, neither of which keys on origin.
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

# Static files shipped with no Cache-Control at all, only an etag — which
# lets a browser serve a stale copy without ever asking. The practical cost
# is that a deploy is invisible: you change a page, load it, and see the old
# one, which is indistinguishable from the change not working. That cost an
# hour of chasing a streaming bug that had already been fixed.
#
# HTML revalidates every time. The etag makes that a 304 with no body, so it
# is close to free and a deploy shows up immediately. Images keep a real
# cache lifetime — they change rarely and are the only heavy thing here.
@app.middleware("http")
async def cache_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".html", "/")) or path.startswith("/demo") and "." not in path.rsplit("/", 1)[-1]:
        response.headers.setdefault("Cache-Control", "no-cache")
    elif path.endswith((".png", ".jpg", ".jpeg", ".svg", ".webp", ".ico")):
        response.headers.setdefault("Cache-Control", "public, max-age=86400")
    elif path.endswith(".js"):
        # Unhashed filename, so it must revalidate too or a shipped fix sits
        # unused in a cache.
        response.headers.setdefault("Cache-Control", "no-cache")
    return response



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


# Which page a bare domain lands on, by hostname. Search and the concierge
# are the same service, so without this every custom domain would land on the
# same page and one of the two tools would only be reachable as a path under
# the other. Matching on the leading label keeps it working for any domain
# pointed at this service rather than hardcoding one.
_HOST_LANDING = {
    "concierge": "/demo/concierge.html",
    "support": "/demo/concierge.html",
    "search": "/demo/podcast.html",
    "tokens": "/demo/assets.html",
    "assets": "/demo/assets.html",
}
_DEFAULT_LANDING = "/demo/podcast.html"


@app.get("/", include_in_schema=False)
async def root(request: Request) -> RedirectResponse:
    # Bare domain -> the page that hostname is for; Market Bubble search is
    # the default, since it is the public entry point.
    host = (request.headers.get("host") or "").split(":")[0].lower()
    label = host.split(".")[0] if "." in host else ""
    return RedirectResponse(url=_HOST_LANDING.get(label, _DEFAULT_LANDING))


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse,
          dependencies=[Depends(public_rate_limit), Depends(global_rate_limit),
                        Depends(daily_budget), Depends(per_client_daily)])
async def chat(
    body: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    agent: ConciergeAgent = Depends(get_agent),
) -> ChatResponse:
    _track("concierge_chats")
    chunks = await retriever.search(body.message, filters=body.filters)
    try:
        response = await agent.answer(
            body.message, body.history, chunks, brief=body.brief
        )
        _classify_outcome(body.message, chunks, response.answer, response.refused)
        return response
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


@app.post("/v1/chat/stream", dependencies=[Depends(public_rate_limit), Depends(global_rate_limit),
                        Depends(daily_budget), Depends(per_client_daily)])
async def chat_stream(
    body: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    agent: ConciergeAgent = Depends(get_agent),
) -> StreamingResponse:
    _track("concierge_chats", stream=True)
    chunks = await retriever.search(body.message, filters=body.filters)

    async def event_source():
        # Sources first so the UI can render citations immediately.
        sources = [
            {"id": c.id, "source_type": c.source_type.value, "score": c.score,
             "metadata": c.metadata}
            for c in chunks
        ]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        parts: list[str] = []
        try:
            async for delta in agent.stream(body.message, body.history, chunks):
                if delta == "\x00REFUSAL\x00":
                    # Whole fallback chain refused mid-stream: the partial
                    # text is invalid — tell the client to replace it.
                    _classify_outcome(body.message, chunks, "", refused=True)
                    payload = json.dumps({"text": REFUSAL_MESSAGE})
                    yield f"event: refusal\ndata: {payload}\n\n"
                    return
                parts.append(delta)
                yield f"data: {json.dumps({'text': delta})}\n\n"
            # Same feedback loop as /v1/chat, on the fully-streamed answer.
            _classify_outcome(body.message, chunks, "".join(parts), refused=False)
            yield "event: done\ndata: {}\n\n"
        except asyncio.CancelledError:
            raise  # client disconnected — let it propagate, don't swallow
        except Exception as exc:  # noqa: BLE001
            # Headers + earlier events are already flushed, so an uncaught
            # error here would leave the client hanging with no terminal
            # event. Always emit event:error so the UI can recover.
            logger.exception("Chat stream failure: %s", exc)
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
          dependencies=[Depends(public_rate_limit), Depends(global_rate_limit),
                        Depends(daily_budget), Depends(per_client_daily)])
async def podcast_search(
    body: PodcastSearchRequest,
    podcast: PodcastIndex = Depends(get_podcast),
) -> PodcastSearchResponse:
    _track("podcast_searches", q=body.query[:120])
    try:
        return await podcast.search(body.query, top_k=body.top_k)
    except anthropic.RateLimitError as exc:
        raise HTTPException(
            status_code=429, detail="Rate limited; retry shortly."
        ) from exc
    except anthropic.APIError as exc:
        status = getattr(exc, "status_code", None)
        logger.error("Anthropic error on search: %s (%s)", type(exc).__name__, status)
        raise HTTPException(status_code=502, detail="Model provider error.") from exc


@app.post("/v1/podcast/search/stream",
          dependencies=[Depends(public_rate_limit), Depends(global_rate_limit),
                        Depends(daily_budget), Depends(per_client_daily)])
async def podcast_search_stream(
    body: PodcastSearchRequest,
    podcast: PodcastIndex = Depends(get_podcast),
) -> StreamingResponse:
    """SSE variant: hits render immediately, the answer streams in."""
    _track("podcast_searches", q=body.query[:120], stream=True)
    hits = await podcast.retrieve(body.query, body.top_k)

    async def event_source():
        payload = json.dumps([h.model_dump() for h in hits])
        yield f"event: hits\ndata: {payload}\n\n"
        try:
            async for delta in podcast.answer_stream(body.query, hits):
                if delta == "\x00REFUSAL\x00":
                    refusal = json.dumps({"text": PODCAST_REFUSAL})
                    yield f"event: refusal\ndata: {refusal}\n\n"
                    return
                yield f"data: {json.dumps({'text': delta})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Podcast stream failure: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': 'stream failed'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/whoami", dependencies=[Depends(require_admin)])
async def whoami(request: Request) -> dict:
    """What this app believes about who is calling. Admin-only.

    Exists because rate limiting behind a proxy cannot be reasoned about from
    documentation — it has to be measured. Which X-Forwarded-For entry is
    trustworthy depends on how many proxies the host actually runs and what
    each one writes, and getting it wrong fails in both directions: read too
    far left and callers pick their own bucket, too far right and everyone
    lands in a bucket keyed on a value that rotates, which quietly switches
    per-IP limiting off entirely.

    Returns the raw header, the resolved key, and the peer address, so the
    chain can be read off a single request instead of inferred.

    Re-gated behind the admin token now that the chain has been read. It is
    caller-scoped so it was never a data leak, but the proxy layout is
    operational detail with no reason to be public.

    The reading it produced, for the record:
        49.36.72.251, 172.69.179.154, 10.192.63.131
        client         Cloudflare       Render
    """
    xff = request.headers.get("x-forwarded-for", "")
    return {
        "x_forwarded_for_raw": xff,
        "hops": [h.strip() for h in xff.split(",") if h.strip()],
        "peer": request.client.host if request.client else None,
        "resolved_rate_limit_key": RateLimiter._client_ip(request),
        "trusted_proxy_hops": get_settings().trusted_proxy_hops,
        "other_ip_headers": {
            k: v for k, v in request.headers.items()
            if k.lower() in ("x-real-ip", "cf-connecting-ip", "true-client-ip",
                             "render-proxy-ip", "fly-client-ip", "forwarded")
        },
    }


@app.get("/v1/stats")
async def stats() -> dict:
    """Usage counters since last restart (durable record: ANALYTICS log lines).

    Includes the daily budget so how close the service is to its ceiling is
    visible without reading logs — a cap you can't see is one you only find
    out about when it starts refusing people.
    """
    return {**STATS, "daily_budget": daily_budget.state(),
            "per_client": per_client_daily.state()}


@app.get("/v1/gaps", dependencies=[Depends(require_admin)])
async def gaps() -> dict:
    """Admin-only feedback loop: the questions the concierge couldn't answer,
    so you know which knowledge-base docs to write next. Requires
    X-Admin-Token when ADMIN_TOKEN is set."""
    top = Counter(g["query"].strip().lower() for g in GAPS).most_common(25)
    return {
        "total_unanswered": STATS["unanswered_chats"],
        "refusals": STATS["refusals"],
        "top_unanswered": [{"query": q, "count": n} for q, n in top],
        "recent": list(GAPS)[-50:],
    }


@app.post("/v1/podcast/ingest", dependencies=[Depends(require_admin)])
async def podcast_ingest(
    episodes: list[Episode],
    podcast: PodcastIndex = Depends(get_podcast),
) -> dict:
    """Admin endpoint — requires X-Admin-Token when ADMIN_TOKEN is set."""
    count = await podcast.ingest(episodes)
    return {"windows_indexed": count}


async def _assets_report(request: Request) -> dict:
    """The aggregated asset report, cached. Shared by the list and detail views."""
    store = request.app.state.assets

    # Short TTL cache: aggregation is pure CPU but the Pinecone fetch isn't.
    cached = getattr(app.state, "_assets_cache", None)
    now = asyncio.get_event_loop().time()
    if cached and now - cached[0] < 300:
        return cached[1]

    try:
        hits = await store.all_hits()
    except Exception as exc:  # noqa: BLE001 — dashboard must not 500
        logger.warning("asset store unavailable (%s); using local file", exc)
        hits = []

    if hits:
        report = aggregate_assets(hits)
        report["episodes_processed"] = len({h.get("episode_id") for h in hits})
    else:
        path = _ROOT / "data" / "assets.json"
        if not path.exists():
            return {"assets": [], "total_hits": 0, "episodes_processed": 0}
        report = json.loads(path.read_text())

    app.state._assets_cache = (now, report)
    return report


@app.get("/v1/assets", dependencies=[Depends(public_rate_limit)])
async def assets(request: Request) -> dict:
    """Assets discussed across the episodes.

    Reads the per-episode hits stored in Pinecone by the weekly sync and
    aggregates them here, so a new episode shows up without a redeploy.
    Falls back to the committed data/assets.json when nothing is stored yet
    (fresh install, or the store hasn't been populated).
    """
    _track("asset_dashboard_views")
    return await _assets_report(request)


@app.get("/v1/assets/{symbol}", dependencies=[Depends(public_rate_limit)])
async def asset_detail(symbol: str, request: Request) -> dict:
    """One asset: what the hosts said about it, plus live Solana market data.

    This is the surface an agent calls to answer "what has Market Bubble said
    about $X?". It reports and cites; it never advises, and the market block
    is present only when the ticker resolves to a Jupiter-verified mint with
    real liquidity. An unresolved ticker yields `market: null`, never a guess.
    """
    ticker = market.clean_symbol(symbol)
    if ticker is None:
        raise HTTPException(status_code=404, detail="Unknown asset.")

    report = await _assets_report(request)
    row = next((a for a in report.get("assets", [])
                if str(a.get("symbol", "")).upper() == ticker), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown asset.")

    _track("asset_detail_views")
    return {
        "symbol": row.get("symbol"),
        "name": row.get("name"),
        "asset_class": row.get("asset_class"),
        "mentions": row.get("mentions"),
        "analysis": row.get("analysis"),
        "moments": row.get("moments", []),
        "market": await _market_for(ticker, row.get("asset_class")),
        "disclaimer": "What was said on the podcast, with timestamps. "
                      "Not advice, not a recommendation, not a price forecast.",
    }


# Symbols CoinGecko cannot price and Jupiter will never list. Skipping them
# saves two upstream round trips per row that can only ever return nothing.
_UNPRICEABLE_CLASSES = {"stock", "index", "commodity", "other"}

_CG_TTL_SECONDS = 600
_MARKET_TTL_SECONDS = 60


async def _coingecko_table() -> dict:
    """The symbol table, refreshed on a TTL. Stale beats empty; never raises."""
    now = asyncio.get_event_loop().time()
    cached = getattr(app.state, "_cg_table", None)
    if cached and now - cached[0] < _CG_TTL_SECONDS:
        return cached[1]
    try:
        table = await market.fetch_coingecko_table()
    except Exception as exc:  # noqa: BLE001
        logger.warning("coingecko table refresh failed: %s", exc)
        return cached[1] if cached else {}
    if not table and cached:
        return cached[1]                 # a failed refresh must not erase data
    app.state._cg_table = (now, table)
    return table


async def _market_for(ticker: str, asset_class: str | None) -> dict | None:
    """Price, and a trade route only where one is earned. Never raises.

    Market data is decoration; the citations are the product. A third party
    being down, slow or wrong must never take out the page that quotes the
    podcast, so every failure path here resolves to None.
    """
    if asset_class in _UNPRICEABLE_CLASSES:
        return None
    cache: dict = app.state._market_cache
    now = asyncio.get_event_loop().time()
    hit = cache.get(ticker)
    if hit and now - hit[0] < _MARKET_TTL_SECONDS:
        return hit[1]
    try:
        table = await _coingecko_table()
        data = await market.quote(ticker, coingecko_table=table)
    except Exception as exc:  # noqa: BLE001
        logger.warning("market lookup failed for %s: %s", ticker, exc)
        data = None
    if len(cache) >= 256:                # bounded: the ticker space is not
        cache.clear()
    cache[ticker] = (now, data)
    return data


@app.get("/v1/podcast/episodes", dependencies=[Depends(public_rate_limit)])
async def podcast_episodes(
    summaries: SummaryStore = Depends(get_summaries),
) -> list[dict]:
    """Pre-computed episode summaries — a Pinecone fetch, no model call."""
    _track("episode_summary_views")
    return await summaries.list_all()
