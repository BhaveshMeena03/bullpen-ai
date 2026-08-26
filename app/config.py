"""Central configuration for the Bullpen Concierge backend.

All secrets and tunables are sourced from the environment (or a local
`.env` file) via pydantic-settings, so nothing sensitive lives in code.
"""

from functools import lru_cache

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
    rerank_candidates: int = 12

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
    # Replies carry no URL. X charges $0.200 for a post containing one against
    # $0.015 without — 13x, or the difference between $36 and $313 a month at
    # fifty mentions a day. The citation (timestamp + episode) is the useful
    # part anyway and the link lives in the bio. Turn this on only if someone
    # else is funding the difference.
    # off | seekable | always
    #
    # A reply carrying a URL costs $0.200 against $0.015 without one,
    # whatever it points at. But the two kinds of link are not worth the
    # same money: a YouTube link carries ?t= and lands on the exact second,
    # while an X broadcast link opens a four-hour video at 0:00 and leaves
    # the reader to scrub. The episode name and the timestamp in the text
    # get them to the same place for a fraction of the price.
    #
    # "seekable" pays only when the link actually jumps — about three
    # answers in eight on this corpus, so roughly 60% off the link bill for
    # nothing anyone would notice missing. "always" is $157/month at the
    # daily cap; "off" is $18; this is around $60.
    x_bot_include_links: str = "seekable"
    # Seconds between polls, jittered. Reads are deduplicated within a UTC
    # day, so frequent polling costs nothing extra; the jitter is about not
    # looking like a metronome, which is a documented suspension trigger.
    x_bot_poll_seconds: float = 60.0
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
    x_bot_verified_only: bool = False
    # Longest reply to compose. 280 is what X API v2 is widely reported to
    # enforce on POST /2/tweets even for Premium accounts — but automated
    # accounts are visibly posting far longer, so one of those is wrong and
    # it is cheap to find out: raise this, send one reply, and either it
    # posts or X answers 400 "Your Tweet text is too long" and costs nothing.
    x_bot_post_limit: int = 280

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
