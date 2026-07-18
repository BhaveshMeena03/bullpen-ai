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
    # Default: Opus 4.8 — strong answers at half the Fable 5 price.
    # Swap via env with no code changes:
    #   ANTHROPIC_MODEL=claude-sonnet-5   cheapest good option ($3/$15)
    #   ANTHROPIC_MODEL=claude-fable-5    max capability ($10/$50)
    # agent.py adapts the request shape per model (thinking config and
    # the Opus fallback are model-specific).
    anthropic_model: str = "claude-opus-4-8"
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

    # --- Voyage AI (embeddings — Anthropic's recommended partner) ----------
    voyage_api_key: str
    voyage_model: str = "voyage-3.5"
    embedding_dimension: int = 1024

    # --- Pinecone -----------------------------------------------------------
    pinecone_api_key: str
    pinecone_index: str = "bullpen-concierge"

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
    rate_limit_rpm: int = 30
    # Requests/minute across ALL clients — the model-spend ceiling.
    global_rate_limit_rpm: int = 120
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
