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

    # --- Retrieval ----------------------------------------------------------
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.30
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
    per_client_daily_budget: int = 200
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


    @field_validator(
        "anthropic_api_key", "voyage_api_key", "pinecone_api_key",
        "admin_token", mode="before",
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
