"""Central configuration for the Bullpen Concierge backend.

All secrets and tunables are sourced from the environment (or a local
`.env` file) via pydantic-settings, so nothing sensitive lives in code.
"""

from functools import lru_cache

from pydantic import field_validator
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
    def _strip_secret(cls, v):
        # A pasted key with a trailing newline/space raises a cryptic
        # ValueError ("control character in headers") deep in the HTTP client.
        # Strip defensively so a mis-pasted env var can never cause it.
        return v.strip() if isinstance(v, str) else v


@lru_cache
def get_settings() -> Settings:
    return Settings()
