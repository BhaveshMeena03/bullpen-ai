"""Central configuration for the Bullpen Concierge backend.

All secrets and tunables are sourced from the environment (or a local
`.env` file) via pydantic-settings, so nothing sensitive lives in code.
"""

from functools import lru_cache

import re
from urllib.parse import urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Anthropic ---------------------------------------------------------
    anthropic_api_key: str
    # Empty means Anthropic direct, which is what every surface does unless
    # this is set. A proxy that speaks the same API — usepod routes
    # claude-haiku-4-5 at $0.40/$2.00 against Anthropic's $1.00/$5.00 — can
    # be put here instead, and only the podcast search reads it. The
    # concierge answers Bullpen support questions and does not go through
    # anybody else's account.
    anthropic_base_url: str = ""
    # Swap via env with no code changes:
    #   ANTHROPIC_MODEL=claude-opus-4-8   stronger reasoning ($5/$25)
    #   ANTHROPIC_MODEL=claude-fable-5    max capability ($10/$50)
    # agent.py adapts the request shape per model (thinking config and
    # the Opus fallback are model-specific).
    # The concierge is RAG-grounded: retrieval does the heavy lifting, so the
    # model's job is to synthesise the retrieved docs and hold the guardrails —
    # not to reason from scratch. Sonnet 5 does that well at a fraction of
    # Opus's cost. (Haiku would be cheaper still, but this bot is customer
    # facing and safety-sensitive — no financial advice, never touch a seed
    # phrase — so the extra guardrail margin is worth the small premium.)
    anthropic_model: str = "claude-sonnet-5"
    # Per-surface override, currently the same model as everything else.
    #
    # This ran on Haiku for a while, on the strength of a benchmark that
    # showed identical accuracy (8/8) and adversarial behaviour (7/7) at a
    # third of the cost. That benchmark asked eight simple factual
    # questions, which is exactly the workload the small model is good at,
    # and it measured the wrong thing.
    #
    # Harder questions separated them. Asked for an agent that "only reads
    # market data and never touches my wallet", Haiku listed token-sniper
    # under skills that are safe to enable. The docs say that skill will
    # "detect and buy new launches in 45ms" — so the answer told someone who
    # had just asked for no wallet access to switch on the skill that buys
    # things. Sonnet put it under Avoid, and also caught marketplace and
    # x402, which Haiku missed entirely. On another question Sonnet noticed
    # the docs state the MCP tool count three different ways and said so,
    # where Haiku picked one and stated it flatly.
    #
    # The economics also stopped favouring it. With the answer cache, a
    # repeated question costs nothing on either model, so the price gap only
    # applies to first-time questions — a shrinking slice. Paying more on a
    # shrinking slice to avoid advice that could cost someone money is not a
    # close call.
    clawpump_model: str = "claude-sonnet-5"
    anthropic_fallback_model: str = "claude-opus-4-8"  # used on Fable 5 only
    # Episode summaries are a one-time batch job per episode; Sonnet 5 is
    # excellent at summarization at 60% less cost than Opus.
    summary_model: str = "claude-sonnet-5"
    # Podcast search answers are 2-3 sentences over a few excerpts — a light
    # task. Haiku 4.5 handles grounded summarization well at ~1/5 the cost
    # of Sonnet, which stretches a small budget across far more queries.
    search_model: str = "claude-haiku-4-5"
    search_effort: str = "low"
    search_max_tokens: int = 1024
    search_timeout_seconds: float = 45.0
    # Adaptive-thinking depth: low | medium | high | xhigh | max
    effort: str = "high"
    max_tokens: int = 16000
    # Ceiling for answers requested with brief=true (chat surfaces). ~120
    # words is well under this; the cap is the backstop, not the target.
    brief_max_tokens: int = 400

    # --- Voyage AI (embeddings — Anthropic's recommended partner) ----------
    voyage_api_key: str
    voyage_model: str = "voyage-3.5"
    # Seconds to wait between embedding batches. 0 on a paid key; set to 21
    # if the account is ever back on the free tier's 3 requests/minute.
    # Rate limits are still handled by the 429 retry in embeddings.py.
    voyage_request_gap_seconds: float = 0.0
    embedding_dimension: int = 1024

    # --- Pinecone -----------------------------------------------------------
    pinecone_api_key: str
    pinecone_index: str = "bullpen-concierge"
    # The MCG archive lives in its own Pinecone index, not a namespace
    # inside this one. That was how it was built -- its ingest refuses to
    # run against the Market Bubble index at all -- and it is the strongest
    # separation of the three corpora: Market Bubble and the Musk
    # interviews share an index and are kept apart by namespace, while MCG
    # cannot reach either of them even by a namespace typo.
    #
    # 11,001 vectors, already embedded. Pointing at them beats re-embedding
    # 444 hours to move them somewhere tidier.
    mcg_pinecone_index: str = "mcg-search"
    mcg_namespace: str = "mcg"
    # Hard ceiling on a single Pinecone write. The SDK's HTTP client has no
    # read timeout, so a half-open socket (seen once: a write hung 2.5h with
    # the connection ESTABLISHED but dead) blocks forever. Bounding the write
    # turns that into a fast failure the ingest's idempotent retry can recover
    # from. A few-hundred-vector upsert takes ~2s, so 60s is generous headroom.
    pinecone_write_timeout_seconds: float = 60.0
    # The same half-open socket hangs a read, and reads are worse: they sit on
    # the request path, and every one runs in a bounded asyncio.to_thread pool.
    # Threads stuck forever exhaust that pool and take down every offloaded
    # call in the process, not just search. Only the writes were bounded when
    # this was first found. A query normally returns in well under a second.
    pinecone_read_timeout_seconds: float = 20.0

    # --- Answer cache --------------------------------------------------------
    # Whole answers, keyed on the question. Support traffic is mostly repeats,
    # and without this the thousandth person to ask pays what the first did.
    #
    # The real invalidation is ingestion, not time: both ingest endpoints
    # clear the cache, so a corrected document takes effect immediately. A
    # cached answer therefore cannot be staler than the index it came from
    # — if the source changed upstream and nothing was re-ingested, the
    # index is wrong too, and expiring the cache only pays to regenerate the
    # same outdated answer.
    #
    # So the TTL is a backstop, not the mechanism, and it started far too
    # short. At a few visitors a day, entries written at 24h expire long
    # before anyone asks again and the cache never pays off. Seven days lets
    # the popular questions actually accumulate hits.
    #
    # 2000 entries is roughly 4MB of answers — nothing, against a service
    # that already holds an embedding client and an HTTP pool. Eviction is
    # least-recently-used, so the ceiling only ever drops questions nobody
    # is asking. Set entries to 0 to disable.
    answer_cache_max_entries: int = 2_000
    answer_cache_ttl_seconds: float = 604_800.0

    # --- Retrieval ----------------------------------------------------------
    retrieval_top_k: int = 6
    # Floor on the RAW vector score, applied before reranking.
    #
    # Measured on this corpus, that score barely separates relevant from
    # irrelevant: "what did se yong park say" — a guest who is genuinely in
    # an indexed episode — scores 0.285, while "recipe for chocolate cake"
    # scores 0.425 and "what is the capital of peru" 0.388. At a floor of
    # 0.30 the real question was dropped and the nonsense sailed through,
    # which is exactly backwards.
    #
    # Cosine similarity over long transcript windows behaves like that:
    # everything is moderately similar to everything, and the spread between
    # a good match and a bad one is smaller than the spread between one
    # phrasing and another. The reranker is the component that actually
    # judges relevance, and it never saw these because the floor ran first.
    #
    # So the floor is now only a guard against a degenerate embedding, and
    # relevance is decided by the reranker and then by the model, which
    # still answers "I couldn't find that" when the excerpts do not support
    # an answer. Verified: nonsense queries still refuse.
    retrieval_min_score: float = 0.05
    # Rerank: pull a wider candidate set from Pinecone, then re-score with
    # Voyage's reranker for actual relevance. Unset RERANK_MODEL to disable.
    rerank_model: str | None = "rerank-2.5-lite"
    # Fifty, not twelve. The reranker is far better at judging
    # relevance than cosine similarity is, and handing it twelve
    # candidates out of twenty-seven thousand passages starves it: the
    # line naming who sold their entire ETH position ranks 44th, so the
    # stage that would recognise it instantly never saw it.
    rerank_candidates: int = 50
    # The reranker runs twice -- once over the fifty above, once over
    # the first twelve of them -- and the model is shown the union. The
    # deep pass reaches passages the shallow one cannot; the shallow pass
    # keeps the answers the deep one evicts, which reranking fifty alone
    # does because positions two to six fill with passages merely about
    # the same subject.
    #
    # Sequencing the two is a no-op, and that was measured rather than
    # assumed: reranking is a total order, so narrowing fifty to twelve
    # and reranking those twelve returns the same six.
    #
    # Across a hundred questions -- twenty of them about topics verified
    # absent from every transcript -- misattributed quotes went 3 at
    # twelve, 2 at fifty, 0 at the union, with the best recall of the
    # three and no answer invented at any setting.
    #
    # Set to 0 to turn the second pass off without a deploy.
    rerank_narrow_pool: int = 12

    # --- Ingestion ----------------------------------------------------------
    chunk_max_chars: int = 2400
    chunk_overlap_chars: int = 240

    # --- API protection ------------------------------------------------------
    # Requests/minute per client IP on public endpoints.
    # Hard ceiling on model-backed requests per UTC day. The per-minute limits
    # stop a burst but not a slow drain: 25/min sits inside every other limit
    # and still reaches ~36,000 requests a day.
    #
    # 3000 is the number because of what it costs, not what it allows. A
    # concierge answer is ~1.4c, so a maxed-out day is about $42 — a bad day,
    # not a bad month. Set to 0 to disable.
    #
    # Raised from 500 ahead of showing this to people who might share it. The
    # cost of getting the cap wrong is asymmetric: too high costs a few tens
    # of dollars once, too low means the people you most wanted to impress get
    # told to come back tomorrow. The log line on exhaustion says when real
    # usage approaches it.
    #
    # This is the ONLY thing bounding spend against a slow drain. The global
    # per-minute limit catches bursts; it does nothing about one client
    # trickling requests all day, which is why the per-IP limit below was cut
    # at the same time this went up.
    #
    # NOTE: render.yaml sets these three as environment variables, which take
    # precedence over everything here. Editing this file alone changes nothing
    # in production — that mistake was made once already.
    # Per-client slice of the daily budget above. 200 is ~5-10x what an
    # enthusiastic person does in a day, so only automation should meet it,
    # and draining the service now needs ~15 distinct addresses rather than
    # one patient script. 0 disables.
    # Raised from 200 after the owner of the site could not use his own demo:
    # a verification run had already spent the allowance, and most of what it
    # spent it on was free — cache hits and retrieval-only checks, which are
    # now refunded (see RateLimiter.refund).
    #
    # Note this does not raise what the service can spend in a day. That
    # ceiling is daily_request_budget below, across all callers. This number
    # only decides how much of it one address may take, so the effect is on
    # fairness, not on the bill.
    per_client_daily_budget: int = 400
    daily_request_budget: int = 3000
    # 12/min per client. A person asks maybe 1-5 questions a minute, so this
    # is still 2-3x human speed and no real user will meet it. It was 30,
    # which is ~10x human and let a single scripted client drain a whole day's
    # budget in about an hour. At 12 that takes over four hours — slow enough
    # to show up in the logs while there is still a day left to save.
    rate_limit_rpm: int = 12
    # Requests/minute across ALL clients. Not the spend ceiling — the daily
    # budget is — this exists so one burst can't outrun the single process.
    # 240/min is ~4/second, comfortably above any organic spike and still far
    # below what would be needed to matter to the daily cap.
    global_rate_limit_rpm: int = 240
    # How many proxies sit in front of this app, used to locate the real
    # client in X-Forwarded-For. Each proxy appends the peer it received from,
    # so the client is this many entries from the right.
    #
    # Two here, measured rather than assumed: Cloudflare is proxying (orange
    # cloud, not DNS-only) and Render's router adds a hop of its own, giving
    # "client, cloudflare, render". Only matters as a fallback — Cloudflare's
    # CF-Connecting-IP is preferred and cannot be forged.
    #
    # Set it too high and the caller's own forged entry gets selected, so
    # change it only alongside a fresh reading from /v1/whoami.
    proxies_in_front: int = 2
    # When set, /v1/ingest and /v1/podcast/ingest require this value in the
    # X-Admin-Token header. Leave unset only for local development.
    admin_token: str | None = None

    # --- X mention bot -----------------------------------------------------
    # Tag the account with a question, it answers from the transcripts. OAuth
    # 1.0a user context, because app-only tokens cannot post.
    x_api_key: str | None = None
    x_api_secret: str | None = None
    x_access_token: str | None = None
    x_access_secret: str | None = None
    # The bot's own numeric id, used for the mentions endpoint and to keep it
    # from answering itself.
    x_bot_user_id: str | None = None
    # Off unless deliberately switched on. The bot spends money on every
    # reply, so it should never start just because credentials happen to be
    # present in the environment.
    x_bot_enabled: bool = False
    # A reply is $0.015 and an answer is about $0.008, so 100 replies is
    # roughly $2.30 a day. The cap is a spend guard: it bounds what a bug, or
    # a raid, can cost before anyone notices.
    x_bot_daily_reply_cap: int = 100
    # Replies this account will put into one conversation in a day.
    # Two automated accounts otherwise answer each other forever.
    x_bot_per_thread_cap: int = 3
    # Replies one ACCOUNT can be given in a day. The per-thread cap does
    # not bound this -- thirty mentions in thirty threads is thirty
    # conversations and no repeats -- and while the bot only answered
    # verified accounts, that gate limited it by accident. Answering
    # everyone removes the accident: without this, one person can take the
    # whole daily cap and leave nothing for anybody else.
    #
    # Priority authors are exempt, same as the daily cap.
    x_bot_per_author_cap: int = 8
    # off | seekable | always. Production runs "always" (render.yaml), which
    # is what decides this — not the default here.
    #
    # There is no cost argument left. X's $0.200 URL surcharge applies to a
    # standalone post, not to a reply, and every mention answer is a reply.
    # An earlier version of this comment claimed 13x on replies and priced
    # three modes off it; the numbers were wrong and the whole compromise
    # they justified was solving nothing.
    #
    # The product argument survives on its own: a YouTube link carries ?t=
    # and lands on the exact second, while an X broadcast link opens a
    # four-hour video at 0:00. "always" ships the X link anyway, because it
    # renders a card with the episode title and thumbnail, and looking like
    # something beats looking like nothing.
    x_bot_include_links: str = "seekable"
    # Seconds between polls, jittered 0.7-1.4x.
    #
    # This is nearly all the latency. Answering takes 4-8 seconds — embed,
    # Pinecone, rerank, model — while a mention waited up to 84 seconds at
    # 60s just to be noticed. Ten times the delay, in the part doing no work.
    #
    # Polling more often is free: X charges per resource returned and
    # deduplicates within the UTC day, so a poll that finds nothing costs
    # nothing. At 20s that is three requests a minute, comfortably inside
    # any published limit, and the client already backs off on a 429.
    #
    # Not lower than that on purpose. A reply landing three seconds after
    # the question reads as a machine, and "reply speed no human could
    # achieve" is a documented suspension trigger. Twenty seconds plus
    # jitter is responsive without being uncanny.
    x_bot_poll_seconds: float = 20.0
    # How long a freshly started instance waits before its first poll.
    # Render's deploy runs the new container alongside the old until the
    # new one is healthy, so for that window two of them poll the same
    # mentions — which is how one question got two different answers five
    # seconds apart. Costs a slower first reply after a deploy; the
    # mention is not lost, because since_id only moves past what was
    # actually answered.
    x_bot_startup_grace_seconds: int = 90
    # Answered from here rather than from retrieval: the contract address
    # is a fact about the project, not something said on the podcast, and
    # it is the one answer that must never be paraphrased or half-right.
    x_bot_contract_address: str | None = None
    # What that address is FOR. A bare "CA: 8VjF..." gets quoted and
    # screenshotted out of context, where 44 characters with no name
    # attached look like any other 44 characters.
    x_bot_token_label: str | None = None
    # The ceiling the reply cap is not. Replies are the expensive part but
    # not the only part: every mention read costs $0.001 whether or not it
    # is answered, and how often the account gets tagged is decided by
    # other people. This bounds a day no matter what they do. 0 disables.
    x_bot_daily_spend_cap_usd: float = 5.0
    # Answer only accounts carrying X's badge. It filters throwaway accounts
    # rather than bad intentions — the badge now means "pays for Premium",
    # not "is who they claim to be" — but a spam account is exactly what it
    # does stop, and every skipped reply saves $0.209 with links on.
    #
    # The cost is the other side: it ignores genuine people who do not pay X
    # for a checkmark, and on a tool whose whole pitch is being useful to
    # whoever asks, that is a real thing to give up. Off by default.
    # An outbound proxy for the clip downloader, when one is needed.
    # YouTube treats a datacenter IP differently from a home one and can
    # refuse a section download from a cloud host while the same URL works
    # on a laptop. Empty means direct, which is the right default: a proxy
    # sees every request, so it is opt-in rather than always on.
    clip_proxy: str = ""
    # Path to a Netscape-format cookies.txt for the clip downloader.
    #
    # YouTube answers an unauthenticated request from a flagged address
    # with "Sign in to confirm you're not a bot", and no amount of retrying
    # or changing player client gets past that; a signed-in session does.
    #
    # It is a secret file rather than a value in the environment on
    # purpose: the file is a live session, and anyone holding it can act as
    # that account. Use a throwaway Google account signed in to nothing
    # else. Google also flags accounts whose cookies appear from datacentre
    # addresses and rotating residential exits, which is exactly what this
    # does, so expect the account to be locked eventually and do not let
    # that matter.
    #
    # Empty means no cookies, which is the right default: the clipper works
    # without them for X, which is most of the archive.
    yt_cookies_file: str = ""

    x_bot_verified_only: bool = False
    # Longest reply to compose. 280 is what X API v2 is widely reported to
    # enforce on POST /2/tweets even for Premium accounts — but automated
    # accounts are visibly posting far longer, so one of those is wrong and
    # it is cheap to find out: raise this, send one reply, and either it
    # posts or X answers 400 "Your Tweet text is too long" and costs nothing.
    x_bot_post_limit: int = 280
    # Summaries get their own, larger budget. They already exist, already
    # carry timestamps, and run to about 3,100 characters — trimming one to
    # fit an answer-sized reply would cut the back half of an episode off.
    x_bot_summary_limit: int = 4000
    # Numeric user ids that must never be met with silence, comma-separated.
    #
    # A stranger getting no reply costs nothing — a weak answer to them is
    # worse than none. One of the hosts getting no reply is different: it
    # reads as a broken tool in front of exactly the people who could put it
    # in front of an audience.
    #
    # So for these, a question that would otherwise produce silence gets a
    # real fact from the archive instead, and the daily cap does not apply.
    # Ids rather than handles because that is what a mention carries, and
    # because a handle can change hands.
    #
    #   948689053            @TheGreatCattsby  co-founder, Market Bubble
    #   363811679            @Banks            host
    #   973261472            @blknoiz06        Ansem, host
    #   1729276822485889024  @MarketBubble     the show
    #   1070767274808696832  @Lexx_eth         runs this account
    x_bot_priority_authors: str = ""
    # Shown when someone asks what this account is. Worth carrying the link
    # here specifically: that question is the one moment where the person
    # asking actually wants somewhere to go.
    x_bot_site: str = "search.lexthedev.com"

    @property
    def priority_author_ids(self) -> set[str]:
        return {p.strip() for p in self.x_bot_priority_authors.split(",")
                if p.strip()}

    @field_validator(
        "anthropic_api_key", "voyage_api_key", "pinecone_api_key",
        "admin_token", "x_api_key", "x_api_secret", "x_access_token",
        "x_access_secret", mode="before",
    )
    @classmethod
    def _sanitize_secret(cls, v):
        # Keys pasted into dashboards pick up invisible junk: trailing
        # newlines (-> ValueError: control character in headers) and
        # zero-width spaces / NBSP (-> UnicodeEncodeError in the HTTP
        # client). Real API keys are printable ASCII with no spaces, so
        # keep exactly that and discard everything else.
        if isinstance(v, str):
            return "".join(ch for ch in v if 0x21 <= ord(ch) <= 0x7E)
        return v

    @model_validator(mode="after")
    def _admin_token_not_blank(self):
        # A whitespace/invisible-only ADMIN_TOKEN sanitizes to "" above, which
        # require_admin would treat as "auth disabled" — silently unguarding
        # the ingest endpoints. Fail closed: a *provided-but-empty* token is
        # a misconfiguration, so refuse to start rather than run open.
        if self.admin_token is not None and self.admin_token == "":
            raise ValueError(
                "ADMIN_TOKEN was set but contains no printable characters "
                "after sanitization. Unset it for local dev, or provide a "
                "real token."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Anywhere but Anthropic gets a placeholder instead of the real key.
#
# usepod authenticates with a token inside the URL path and ignores the
# Authorization header entirely, so sending the real key there would hand it
# to a third party for nothing. Whether that is where we are going is decided
# on the PARSED HOSTNAME and an exact match: "api.anthropic.com" is a
# substring of api.anthropic.com.evil.example, and a substring test would send
# the key straight to it.
_PLACEHOLDER_KEY = "unused-the-proxy-authenticates-by-url"


def anthropic_client_kwargs(settings: Settings) -> dict:
    """Constructor arguments for an Anthropic client, direct or proxied."""
    base = (settings.anthropic_base_url or "").strip()
    if not base:
        return {"api_key": settings.anthropic_api_key}
    host = (urlparse(base).hostname or "").lower()
    going_direct = host == "api.anthropic.com"
    return {
        "base_url": base,
        "api_key": settings.anthropic_api_key if going_direct
        else _PLACEHOLDER_KEY,
    }


def redact(text: str) -> str:
    """Hide a proxy token in anything about to be printed or logged.

    The token is a path segment, so it turns up in exception text, in stack
    traces and in httpx's own request repr. Redacting BEFORE truncating
    matters: cutting a URL to eighty characters can leave the token intact
    and drop the part that made it look like a URL.
    """
    return re.sub(r"(/proxy/)[^/\s]+", r"\1<token>", text or "")
