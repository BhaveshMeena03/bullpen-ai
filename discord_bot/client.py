"""HTTP client to the Market Bubble Search backend.

The bot is a thin client — all search/RAG logic lives in the deployed API,
so the bot needs only a Discord token and the backend URL (no Anthropic /
Voyage / Pinecone keys). This module is pure and unit-testable: no Discord
imports, all network via an injectable httpx client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("mbbot.client")


class SearchError(Exception):
    """Search failed in a way worth showing the user a specific message."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind          # "busy" | "unavailable" | "timeout" | "error"
        self.message = message


@dataclass
class Hit:
    timestamp: str
    title: str
    deep_link: str


@dataclass
class SearchResult:
    answer: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False


class SearchClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        # Injectable for tests; owns one in production.
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, *, retries: int = 1) -> SearchResult:
        """POST the query to /v1/podcast/search. Retries once on a transient
        (timeout / 5xx / connection) failure; maps everything else to a
        SearchError with a user-facing kind."""
        url = f"{self._base}/v1/podcast/search"
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = await self._client.post(
                    url, json={"query": query}, timeout=self._timeout
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                logger.warning("search transport error (try %d): %s", attempt, exc)
                continue  # retry
            if resp.status_code == 200:
                return _parse(resp.json())
            if resp.status_code == 503:
                raise SearchError("busy", "The search is busy right now — "
                                          "give it a few seconds and try again.")
            if resp.status_code == 429:
                raise SearchError("busy", "Too many searches at once — "
                                          "try again in a moment.")
            if 500 <= resp.status_code < 600:
                last_exc = SearchError("unavailable", f"backend {resp.status_code}")
                continue  # retry 5xx
            # 4xx (bad request etc.) — don't retry.
            raise SearchError("error", "Couldn't run that search.")
        if isinstance(last_exc, SearchError):
            raise SearchError("unavailable",
                              "Search is temporarily unavailable — try again soon.")
        raise SearchError("timeout",
                          "Search took too long to respond — try again.")


def _parse(data: dict) -> SearchResult:
    hits = [
        Hit(
            timestamp=str(h.get("timestamp", "")),
            title=str(h.get("title", "")),
            deep_link=str(h.get("deep_link", "")),
        )
        for h in (data.get("hits") or [])
        if h.get("deep_link")
    ]
    return SearchResult(
        answer=str(data.get("answer", "")).strip(),
        hits=hits,
        refused=bool(data.get("refused", False)),
    )
