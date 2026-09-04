"""API layer — asynchronous FastAPI server for the Bullpen Concierge.

Endpoints:
    POST /v1/chat         -> full JSON response with sources + usage
    POST /v1/chat/stream  -> Server-Sent Events token stream
    POST /v1/ingest       -> (internal) push documents into the vector DB
    GET  /healthz         -> liveness probe
"""

import asyncio
import contextlib
import gzip
import json
import logging
import os
import random
import re
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from html import escape as html_escape
from pathlib import Path
from urllib.parse import quote_plus

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from voyageai import error as voyage_error

from . import market, og_card
from .agent import REFUSAL_MESSAGE, ConciergeAgent
from .answer_cache import AnswerCache, make_key
from .assets import aggregate as aggregate_assets
from .assets_store import AssetStore
from .clawpump import NAMESPACE as CLAWPUMP_NAMESPACE
from .clawpump import ClawPumpAgent
from .dedupe import canonical_episode_ids
from .clipper import (
    MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    ClipService,
    ffmpeg_available,
)
from .config import get_settings
from .ingest import IngestionPipeline
from .podcast import REFUSAL_ANSWER as PODCAST_REFUSAL
from .podcast import PodcastIndex
from .questions import QuestionLog
from .retriever import Retriever
from .schemas import (
    ClipRequest,
    ChatRequest,
    ChatResponse,
    Episode,
    IngestDocument,
    PodcastSearchRequest,
    PodcastSearchResponse,
    RetrievedChunk,
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
from .usage import UsageLedger, writable_path

logging.basicConfig(level=logging.INFO)


class _RedactProxyToken(logging.Filter):
    """Keep the inference proxy's token out of the logs.

    usepod authenticates with a token in the URL PATH, and httpx logs every
    request line at INFO — so the first live request wrote a working
    credential into Render's log history, where it stays. Anything holding
    it can spend the balance.

    Filtering here rather than silencing httpx: the request lines are how
    you tell which route answered, and a filter catches whatever else
    decides to log a URL later. Applied to the root handler so it covers
    every logger in the process, and to the record's args as well as its
    message, because "%s" formatting keeps the URL in args until render.
    """

    _TOKEN = re.compile(r"(/proxy/)[^/\s\"']+")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str) and "/proxy/" in record.msg:
            record.msg = self._TOKEN.sub(r"\1<token>", record.msg)
        if record.args:
            args = record.args
            if isinstance(args, dict):
                record.args = {
                    k: (self._TOKEN.sub(r"\1<token>", v)
                        if isinstance(v, str) and "/proxy/" in v else v)
                    for k, v in args.items()}
            else:
                record.args = tuple(
                    self._TOKEN.sub(r"\1<token>", a)
                    if isinstance(a, str) and "/proxy/" in a else a
                    for a in args)
        return True


for _handler in logging.getLogger().handlers:
    _handler.addFilter(_RedactProxyToken())
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


async def _run_x_bot(app: FastAPI, settings) -> None:
    """Poll mentions forever, answering what is new.

    Runs beside the web app and must never take it down with it, so every
    error is caught and the loop keeps its cadence. The two it names are the
    ones that are expected rather than exceptional: an empty credit balance
    is a fact about billing, and a cancelled task is a shutdown.
    """
    from app.x_api import OutOfCreditsError, XClient, XCredentials
    from app.x_bot import STATE_PATH, MentionBot, load_highlights

    beat = app.state.x_bot_heartbeat
    beat.enabled = True
    beat.poll_seconds = settings.x_bot_poll_seconds
    beat.highlights = len(load_highlights())

    bot = MentionBot(
        XClient(
            XCredentials(settings.x_api_key, settings.x_api_secret,
                         settings.x_access_token, settings.x_access_secret),
            bot_user_id=settings.x_bot_user_id,
        ),
        app.state.podcast,
        daily_reply_cap=settings.x_bot_daily_reply_cap,
        per_thread_cap=settings.x_bot_per_thread_cap,
        include_links=settings.x_bot_include_links,
        contract_address=settings.x_bot_contract_address,
        token_label=settings.x_bot_token_label,
        daily_spend_cap_usd=settings.x_bot_daily_spend_cap_usd,
        verified_only=settings.x_bot_verified_only,
        per_author_cap=settings.x_bot_per_author_cap,
        post_limit=settings.x_bot_post_limit,
        summary_limit=settings.x_bot_summary_limit,
        summaries=SummaryStore(),
        questions=QuestionLog(),
        priority_authors=settings.priority_author_ids,
        site=settings.x_bot_site,
    )
    # Says whether it can actually work, not just that it started. Both of
    # the silent outages — a state file it could not write, and a highlight
    # pool that never reached the image — would have been one glance here.
    logger.info("x_bot: build %s · cap %d/day · links %s",
                beat.version, settings.x_bot_daily_reply_cap,
                settings.x_bot_include_links)
    logger.info("x_bot:   state file : %s", STATE_PATH)
    logger.info("x_bot:   highlights : %d loaded", beat.highlights)
    logger.info("x_bot:   verified   : %s",
                "badged accounts only" if settings.x_bot_verified_only
                else "everyone")

    # Stand back before the first poll, because for part of a deploy there
    # are two of us.
    #
    # Render brings the new instance up and only signals the old one once
    # this one is healthy, so for that window both are polling the same
    # mentions. The guard in _answer re-reads the timeline immediately
    # before posting, but it cannot help when both instances check before
    # either has posted: @gvgnft asked one question on 1 September and got
    # two different answers five seconds apart, mid-deploy.
    #
    # Waiting here costs a slower first reply after a deploy — the mention
    # is still there, since since_id only advances past what was answered —
    # and removes the overlap that causes it.
    grace = settings.x_bot_startup_grace_seconds
    if grace:
        logger.info("x_bot: holding %ds before the first poll, so a "
                    "deploy's outgoing instance is gone first", grace)
        await asyncio.sleep(grace)

    while True:
        try:
            posted = await bot.tick(time.strftime("%Y-%m-%d", time.gmtime()))
            beat.polled()
            beat.posted(posted)
            if posted:
                logger.info("x_bot: posted %d repl%s today=%d",
                            posted, "y" if posted == 1 else "ies",
                            bot.state.replies_today)
        except asyncio.CancelledError:
            raise
        except OutOfCreditsError as exc:
            # Backs off rather than exiting: topping up should not need a
            # redeploy to be noticed.
            logger.error("x_bot: %s — retrying in an hour", exc)
            await asyncio.sleep(3600)
            continue
        except Exception:                                   # noqa: BLE001
            beat.failed()
            logger.exception("x_bot: poll failed (%d in a row) — continuing",
                             beat.consecutive_errors)
        # Jittered, so two instances that do overlap drift apart instead of
        # polling in lockstep. Two loops started seconds apart stay seconds
        # apart forever on a fixed cadence, which is exactly the condition
        # that let both of them read, compose and post inside the same five
        # seconds.
        await asyncio.sleep(
            MentionBot.pause_seconds(settings.x_bot_poll_seconds)
            * random.uniform(0.85, 1.15))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build heavyweight clients once, at startup, and share them.
    app.state.retriever = Retriever()
    # Next to the data files when that is writable, the temp directory when
    # it is not (it is not, in the container). Either way a redeploy clears
    # it — the ANALYTICS usage log lines are the durable record.
    app.state.usage = UsageLedger(
        path=writable_path(_ROOT / "data" / ".usage.json"))
    # Only where the binaries exist. On an image without ffmpeg the
    # endpoints answer 503 rather than accepting a job that can never run.
    app.state.clips = (ClipService(proxy=get_settings().clip_proxy or None)
                       if ffmpeg_available() else None)
    app.state.agent = ConciergeAgent(ledger=app.state.usage)
    app.state.clawpump_agent = ClawPumpAgent(ledger=app.state.usage)
    app.state.pipeline = IngestionPipeline()
    app.state.podcast = PodcastIndex(ledger=app.state.usage)
    app.state.summaries = SummaryStore()
    app.state.assets = AssetStore()
    _s = get_settings()
    # One cache, shared by every surface. The key carries the surface name,
    # so sharing the store cannot leak an answer between knowledge bases.
    app.state.answers = AnswerCache(
        max_entries=_s.answer_cache_max_entries,
        ttl_seconds=_s.answer_cache_ttl_seconds,
    )
    # Bounded per-ticker cache for live market lookups, and the CoinGecko
    # symbol table. Both fill lazily — startup must not wait on a third party.
    app.state._market_cache = {}
    app.state._cg_table = None

    # Created whether or not the bot runs, so "switched off" and "died" are
    # distinguishable from outside. Render exports the commit it built, and
    # reporting it is the one signal that would have caught the outage where
    # a failed build left the previous image serving happily.
    from app.x_bot import Heartbeat
    app.state.x_bot_heartbeat = Heartbeat(
        version=(os.environ.get("RENDER_GIT_COMMIT") or "local")[:7],
        started_at=time.time(),
    )

    # The X mention bot, in this process rather than a service of its own.
    # It shares the PodcastIndex already built above, which is the point:
    # going over HTTP would put it behind this app's own per-client limiter
    # and have it compete with real visitors for the daily budget.
    #
    # Off unless X_BOT_ENABLED, so credentials present in the environment are
    # never on their own enough to start replying in public.
    app.state.x_bot_task = None
    if _s.x_bot_enabled:
        app.state.x_bot_task = asyncio.create_task(_run_x_bot(app, _s))
        # Says whether it can actually work, not just that it started.
        # Both of today's silent outages — a state file it could not write
        # and a highlight pool that was never copied into the image — would
        # have been one glance at these lines instead of an afternoon.
        from app.x_bot import STATE_PATH, load_highlights
        logger.info("X mention bot started (cap %d/day, links %s)",
                    _s.x_bot_daily_reply_cap, _s.x_bot_include_links)
        logger.info("  state file : %s", STATE_PATH)
        logger.info("  highlights : %d loaded", len(load_highlights()))
        logger.info("  priority   : %d account(s)",
                    len(_s.priority_author_ids))
    try:
        yield
    finally:
        task = app.state.x_bot_task
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


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
# Before the /demo mount, so a shared search gets a card that names the
# question. X's crawler does not run JavaScript, so the page it fetches is
# the page as written — every search link would otherwise show the same
# generic "search every episode by meaning" card, and the hardcoded og:url
# told X the canonical page was the empty homepage, which is where clicking
# the card actually landed.
@app.get("/og/search.png", include_in_schema=False)
async def og_search_card(request: Request, q: str = ""):
    """The share card, drawn around the question that was actually asked.

    The static PNG had an example question printed in its search box, so
    sharing a search showed the reader two different questions -- the one
    baked into the image and the real one X overlays from og:title.

    Cached hard and keyed by an etag over the question: a crawler fetches
    each card once, and the same link shared twice costs nothing to draw
    again.
    """
    asked = (q or "").strip()[:180]
    tag = f'W/"{og_card.etag(asked)}"'
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304, headers={"ETag": tag})
    try:
        png = await asyncio.to_thread(og_card.render, asked or None)
    except Exception:                                       # noqa: BLE001
        # A card is decoration; a search page that 500s because it could
        # not draw one is not. Fall back to the static image.
        logger.exception("og card render failed")
        return RedirectResponse("/demo/og-broadcast.png", status_code=302)
    return Response(png, media_type="image/png", headers={
        "ETag": tag,
        "Cache-Control": "public, max-age=86400, s-maxage=604800",
    })


@app.get("/demo/podcast.html", include_in_schema=False)
async def podcast_page(request: Request):
    page = (_ROOT / "demo" / "podcast.html").read_text()
    asked = (request.query_params.get("q") or "").strip()
    # The scheme the VISITOR used, not the one this process was handed.
    # Render terminates TLS at its proxy, so request.url is http:// even
    # though every real visitor arrives on https — and a canonical URL on
    # the wrong scheme is a different URL to a crawler.
    canonical = str(request.url)
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    if forwarded == "https" and canonical.startswith("http://"):
        canonical = "https://" + canonical[len("http://"):]
    if asked:
        # The question, as the title. Escaped because it lands inside an
        # HTML attribute and arrives from a URL anybody can craft.
        shown = html_escape(asked[:110], quote=True)
        page = page.replace(
            '<meta property="og:title" content="Market Bubble Search '
            '— ask the broadcast anything">',
            f'<meta property="og:title" content="&#8220;{shown}&#8221;">')
        page = page.replace(
            '<meta name="twitter:title" content="Market Bubble Search '
            '— ask the broadcast anything">',
            f'<meta name="twitter:title" content="&#8220;{shown}&#8221;">')
        # The card is drawn per question too, so the image and the title
        # stop showing two different ones.
        card = f"{canonical.split('?')[0].rsplit('/demo/', 1)[0]}/og/search.png?q={quote_plus(asked[:180])}"
        for tag_name in ("og:image", "twitter:image"):
            prefix = "property" if tag_name.startswith("og:") else "name"
            page = re.sub(
                rf'<meta {prefix}="{tag_name}" content="[^"]*">',
                f'<meta {prefix}="{tag_name}" content="{html_escape(card, quote=True)}">',
                page, count=1)
        answer_line = ("Answered from the Market Bubble transcripts, with "
                       "the moment it was said.")
        for tag in ("og:description", "twitter:description"):
            prefix = "property" if tag.startswith("og:") else "name"
            page = page.replace(
                f'<meta {prefix}="{tag}" content="Search every episode by '
                f'meaning and jump to the exact moment on YouTube. Free, no '
                f'login, open source.">',
                f'<meta {prefix}="{tag}" content="{answer_line}">')
    # Never let og:url override the URL somebody actually shared: it made
    # every search link canonicalise to the empty homepage.
    page = re.sub(r'<meta property="og:url" content="[^"]*">',
                  f'<meta property="og:url" content="{html_escape(canonical, quote=True)}">',
                  page, count=1)
    return HTMLResponse(page)


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


def get_clawpump_agent(request: Request) -> ClawPumpAgent:
    return request.app.state.clawpump_agent


def get_answers(request: Request) -> AnswerCache:
    return request.app.state.answers


def get_usage(request: Request) -> UsageLedger:
    return request.app.state.usage


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


# Admin-gated. It reports what the service costs to run, how many calls each
# surface takes and which models answer them — operational detail about a
# business, sitting open to anyone who guessed the path. Nothing here is a
# credential, which is why it was public; that is not the same as it being
# nobody else's business.
@app.get("/v1/usage", dependencies=[Depends(require_admin)])
async def usage_report(usage: UsageLedger = Depends(get_usage)) -> dict:
    """Model spend per day and per surface, priced from published rates.

    An estimate, not a mirror of the Anthropic console — there is no API for
    an account balance. It is derived from the token counts on real
    responses rather than from request counts, so it is close.
    """
    return usage.report()


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/x-bot/status")
async def x_bot_status(request: Request) -> dict:
    """Whether the mention bot is alive, and which build is running.

    Deliberately unauthenticated and free of anything sensitive: the point
    of this endpoint is that checking it is cheap enough to be automated,
    and a check nobody can run without a token is a check nobody runs. It
    carries no credentials, no user data and no file paths — counters, a
    commit hash, and how long since the loop last completed a cycle.
    """
    beat = getattr(request.app.state, "x_bot_heartbeat", None)
    if beat is None:
        return {"enabled": False, "healthy": True, "version": "unknown",
                "note": "no heartbeat — app started without one"}
    return beat.report()


@app.post("/v1/chat", response_model=ChatResponse,
          dependencies=[Depends(public_rate_limit), Depends(global_rate_limit),
                        Depends(daily_budget), Depends(per_client_daily)])
async def chat(
    body: ChatRequest,
    request: Request,
    retriever: Retriever = Depends(get_retriever),
    agent: ConciergeAgent = Depends(get_agent),
    answers: AnswerCache = Depends(get_answers),
    usage: UsageLedger = Depends(get_usage),
) -> ChatResponse:
    _track("concierge_chats")
    key = make_key(body.message, surface="concierge", brief=body.brief) \
        if _cacheable(body) and not body.filters else None
    if key:
        cached = answers.get(key)
        if cached is not None:
            usage.record("concierge", cached.model, None, cached=True)
            per_client_daily.refund(request)
            return cached
    chunks = await retriever.search(body.message, filters=body.filters)
    try:
        response = await agent.answer(
            body.message, body.history, chunks, brief=body.brief
        )
        _classify_outcome(body.message, chunks, response.answer, response.refused)
        if key:
            answers.put(key, response)
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


@app.post("/v1/clawpump/chat", response_model=ChatResponse,
          dependencies=[Depends(public_rate_limit), Depends(global_rate_limit),
                        Depends(daily_budget), Depends(per_client_daily)])
async def clawpump_chat(
    body: ChatRequest,
    request: Request,
    retriever: Retriever = Depends(get_retriever),
    agent: ClawPumpAgent = Depends(get_clawpump_agent),
    answers: AnswerCache = Depends(get_answers),
    usage: UsageLedger = Depends(get_usage),
) -> ChatResponse:
    """Support answers grounded ONLY in ClawPump's documentation.

    Shares the retriever with the Bullpen concierge — same index, same
    embedding model — and separates the two by namespace. `body.filters` is
    deliberately not forwarded: on this route the caller does not get to
    influence what is searched, because the one thing that must never happen
    here is answering from another product's docs.
    """
    _track("concierge_chats")
    key = make_key(body.message, surface=CLAWPUMP_NAMESPACE, brief=body.brief) \
        if _cacheable(body) else None
    if key:
        cached = answers.get(key)
        if cached is not None:
            usage.record("clawpump-support", cached.model, None, cached=True)
            per_client_daily.refund(request)
            return cached
    chunks = await retriever.search(body.message, namespace=CLAWPUMP_NAMESPACE)
    try:
        response = await agent.answer(
            body.message, body.history, chunks, brief=body.brief
        )
        _classify_outcome(body.message, chunks, response.answer, response.refused)
        if key:
            answers.put(key, response)
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
    return _sse(_chat_event_source(agent, body, chunks))


def _cacheable(body: ChatRequest) -> bool:
    """Only single-turn questions are cacheable.

    A follow-up ("what about for perps?") means nothing without the turns
    before it, so its text is not an identity. Keying those would serve one
    conversation's answer into another's.
    """
    return not body.history


def _sse(source) -> StreamingResponse:
    return StreamingResponse(
        source,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _chat_event_source(agent: ConciergeAgent, body: ChatRequest,
                       chunks: list[RetrievedChunk]):
    """The SSE body for a chat answer, shared by every support surface.

    Extracted rather than copied: the refusal handling, the disconnect case
    and the guarantee that a terminal event is always emitted are the parts
    that took the longest to get right here, and a second surface with its
    own near-copy is where they quietly drift apart.
    """

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

    return event_source()


@app.post("/v1/clawpump/chat/stream",
          dependencies=[Depends(public_rate_limit), Depends(global_rate_limit),
                        Depends(daily_budget), Depends(per_client_daily)])
async def clawpump_chat_stream(
    body: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    agent: ClawPumpAgent = Depends(get_clawpump_agent),
) -> StreamingResponse:
    _track("concierge_chats", stream=True)
    chunks = await retriever.search(body.message, namespace=CLAWPUMP_NAMESPACE)
    return _sse(_chat_event_source(agent, body, chunks))


@app.post("/v1/ingest", dependencies=[Depends(require_admin)])
async def ingest(
    docs: list[IngestDocument],
    pipeline: IngestionPipeline = Depends(get_pipeline),
    answers: AnswerCache = Depends(get_answers),
) -> dict:
    """Admin endpoint — requires X-Admin-Token when ADMIN_TOKEN is set."""
    count = await pipeline.ingest(docs)
    # New documentation makes every cached answer potentially wrong, and the
    # whole reason to fix a doc is that something was wrong. Waiting out a
    # 24h TTL would mean the correction is invisible for a day to exactly
    # the popular questions the cache holds.
    dropped = answers.clear()
    return {"chunks_upserted": count, "cached_answers_dropped": dropped}


@app.post("/v1/podcast/search", response_model=PodcastSearchResponse,
          dependencies=[Depends(public_rate_limit), Depends(global_rate_limit),
                        Depends(daily_budget), Depends(per_client_daily)])
async def podcast_search(
    body: PodcastSearchRequest,
    request: Request,
    podcast: PodcastIndex = Depends(get_podcast),
    answers: AnswerCache = Depends(get_answers),
    usage: UsageLedger = Depends(get_usage),
) -> PodcastSearchResponse:
    _track("podcast_searches", q=body.query[:120])
    # The highest-traffic surface, and the most repetitive: the page ships
    # example chips, and a link that gets shared sends everyone who clicks
    # it to the same query.
    key = make_key(body.query, surface="podcast", top_k=body.top_k)
    cached = answers.get(key)
    if cached is not None:
        usage.record("market-bubble-search", cached.model, None, cached=True)
        # A cache hit costs nothing, so it must not spend the caller's
        # daily allowance. See RateLimiter.refund.
        per_client_daily.refund(request)
        return cached
    try:
        result = await podcast.search(body.query, top_k=body.top_k)
        answers.put(key, result)
        return result
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
            "per_client": per_client_daily.state(),
            "answer_cache": app.state.answers.state()}


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
    answers: AnswerCache = Depends(get_answers),
) -> dict:
    """Admin endpoint — requires X-Admin-Token when ADMIN_TOKEN is set."""
    count = await podcast.ingest(episodes)
    # A new episode changes what the right answer is — most obviously for
    # "what did they say most recently about X".
    dropped = answers.clear()
    return {"windows_indexed": count, "cached_answers_dropped": dropped}


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


_CANONICAL_CACHE: set[str] | None = None


def _canonical_ids() -> set[str]:
    """One episode id per show, computed once.

    The index holds a file per upload, and one broadcast reaches it up to
    four times: the live X post, the YouTube cut, and shorter clips of
    both. Retrieval wants all of them -- the cut drops guest interviews
    and some segments only aired live -- but a list of episodes wants one
    row per evening. The page was rendering 38 cards under a header that
    said 18, with 20 August appearing three times.

    Empty on any failure, which the caller reads as "list everything".
    Showing a duplicate row is a blemish; hiding real episodes because a
    file would not parse is a broken page.
    """
    global _CANONICAL_CACHE
    if _CANONICAL_CACHE is not None:
        return _CANONICAL_CACHE
    try:
        episodes = list(_episodes_by_id().values())
        _CANONICAL_CACHE = canonical_episode_ids(episodes) if episodes else set()
        if _CANONICAL_CACHE:
            logger.info("%d files group into %d shows for the episode list",
                        len(episodes), len(_CANONICAL_CACHE))
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not group episodes into shows (%s) — listing "
                       "every file", exc)
        _CANONICAL_CACHE = set()
    return _CANONICAL_CACHE


def _listed(rows: list[dict]) -> list[dict]:
    """The rows the episode list shows: one per broadcast.

    Only a KNOWN duplicate is hidden. An episode that episodes.json has
    never heard of cannot be judged a duplicate of anything, so it shows.

    That distinction is the whole guard. Filtering to `canonical` alone
    hid the newest episode the moment one was ingested: summaries live in
    Pinecone and are written by the ingest, episodes.json ships with the
    image, so between an ingest and the next deploy the newest show is in
    the index, searchable and answering questions, and absent from the
    list of episodes. It happened within hours of that filter shipping.
    """
    canonical = _canonical_ids()
    if not canonical:
        return rows
    known = set(_episodes_by_id())
    shown = [r for r in rows
             if r.get("episode_id") not in known
             or r.get("episode_id") in canonical]
    return shown or rows


@app.get("/v1/podcast/archive", dependencies=[Depends(public_rate_limit)])
async def podcast_archive(
    summaries: SummaryStore = Depends(get_summaries),
) -> dict:
    """How big the archive is, for the line the page opens with.

    Derived from the same rows the episode list renders, so the headline
    figure and the list below it cannot disagree. They did: the header
    said 18 while the list showed 38, because one was counting shows and
    the other was counting files.

    Hours come from episodes.json, which ships with the image, so a show
    ingested since the last deploy is counted in `shows` and contributes
    nothing to `hours` until it redeploys. Undercounting the hours for a
    few hours is the right way round -- the alternative is a figure that
    claims audio the answer engine cannot quote.
    """
    try:
        rows = _listed(await summaries.list_all())
        episodes = _episodes_by_id()
        seconds = 0.0
        for row in rows:
            episode = episodes.get(row.get("episode_id"))
            if episode:
                seconds += max((s.get("t", 0)
                                for s in episode.get("segments") or []),
                               default=0)
        return {"shows": len(rows), "hours": round(seconds / 3600)}
    except Exception as exc:                                    # noqa: BLE001
        # The page keeps the figures already written into the markup.
        logger.warning("could not size the archive: %s", exc)
        raise HTTPException(status_code=503, detail="unavailable")


@app.get("/v1/podcast/episodes", dependencies=[Depends(public_rate_limit)])
async def podcast_episodes(
    summaries: SummaryStore = Depends(get_summaries),
) -> list[dict]:
    """Pre-computed episode summaries — a Pinecone fetch, no model call.

    One row per show. The duplicates stay in the index and stay
    searchable; they simply do not each get a card.
    """
    _track("episode_summary_views")
    return _listed(await summaries.list_all())


# ─── clips ────────────────────────────────────────────────────────────────
#
# Half the archive is X broadcasts, and X cannot link to a timestamp. For
# those moments a rendered clip is not a nicety, it is the only way anyone
# can point at what was said — which is why this renders at the quality of
# a clip cut by hand rather than at a preview quality.
#
# The cost of that is real: ~2.2 CPU-seconds per second of output, one
# encode at a time. So the limiter here is deliberately far tighter than
# the one on search, and it is per-IP rather than global, because the
# failure being prevented is one person queueing thirty clips.

# Three an hour. A person clipping a moment they just found does it once or
# twice; thirty in a row is a script.
clip_rate_limit = RateLimiter(rpm=3, burst=3)

_EPISODES_CACHE: dict[str, dict] | None = None


def _episodes_by_id() -> dict[str, dict]:
    """episodes.json, parsed once, on the first clip anyone asks for.

    Loaded lazily rather than at boot because it is the largest file the
    service owns and nothing else at runtime reads it: a deploy that never
    cuts a clip should never pay for it.
    """
    global _EPISODES_CACHE
    if _EPISODES_CACHE is not None:
        return _EPISODES_CACHE
    # The plain file locally, the gzipped one in the image. episodes.json is
    # 7.3MB and regenerated by every ingest, so committing it would add a
    # fresh 7MB blob to git each time — it stays ignored, and the 2.3MB
    # gzip beside it is what ships. Decompressing costs 0.04s, once.
    raw = _ROOT / "data" / "episodes.json"
    packed = _ROOT / "data" / "episodes.json.gz"
    try:
        if raw.exists():
            data = json.loads(raw.read_text())
        else:
            with gzip.open(packed, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        _EPISODES_CACHE = {e["episode_id"]: e for e in data}
        logger.info("loaded %d episodes for clipping from %s",
                    len(_EPISODES_CACHE),
                    raw.name if raw.exists() else packed.name)
    except Exception as exc:  # noqa: BLE001
        # Clipping degrades to a 404 per episode rather than taking the
        # service down; everything else here works without this file.
        logger.warning("could not load episodes for clipping: %s", exc)
        _EPISODES_CACHE = {}
    return _EPISODES_CACHE


@app.post("/v1/podcast/clip", dependencies=[Depends(clip_rate_limit)])
async def podcast_clip(req: ClipRequest, request: Request) -> dict:
    """Queue a clip. Returns a job id to poll — the render outlives a request."""
    service = getattr(request.app.state, "clips", None)
    if service is None or not ffmpeg_available():
        raise HTTPException(
            status_code=503,
            detail="Clipping is not available on this server.")

    episode = _episodes_by_id().get(req.episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="No such episode.")

    start, end = float(req.start), float(req.end)
    length = end - start
    if length < MIN_CLIP_SECONDS or length > MAX_CLIP_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"A clip has to be between {MIN_CLIP_SECONDS} and "
                   f"{MAX_CLIP_SECONDS} seconds — that one is "
                   f"{length:.0f}.")

    # A queue this deep already means a wait of minutes; taking the job
    # anyway would just hide that behind a spinner.
    if service.queued_count() >= 4:
        raise HTTPException(
            status_code=429,
            detail="A few clips are already rendering — try again shortly.",
            headers={"Retry-After": "120"})

    job = service.submit(episode, start, end)
    _track("clips_requested")
    return {"job_id": job.id, "status": job.status,
            "seconds": round(length, 1),
            "queue_position": service.queued_count()}


@app.get("/v1/podcast/clip/{job_id}")
async def podcast_clip_status(job_id: str, request: Request) -> dict:
    service = getattr(request.app.state, "clips", None)
    job = service.get(job_id) if service else None
    if job is None:
        # Expired and never-existed are the same answer on purpose: a job id
        # is a capability, and confirming which ids were real would let one
        # be guessed.
        raise HTTPException(status_code=404, detail="No such clip.")
    return {"job_id": job.id, "status": job.status, "error": job.error,
            "ready": job.status == "done",
            "url": f"/v1/podcast/clip/{job.id}/file"
                   if job.status == "done" else None}


@app.get("/v1/podcast/clip/{job_id}/file")
async def podcast_clip_file(job_id: str, request: Request):
    service = getattr(request.app.state, "clips", None)
    job = service.get(job_id) if service else None
    if job is None or job.status != "done" or not job.path:
        raise HTTPException(status_code=404, detail="No such clip.")
    if not job.path.exists():
        # The sweep deletes on a TTL, so a job can be "done" and its file
        # already gone.
        raise HTTPException(status_code=410, detail="That clip has expired.")
    _track("clips_downloaded")
    return FileResponse(job.path, media_type="video/mp4",
                        filename=f"market-bubble-{job.id}.mp4")
