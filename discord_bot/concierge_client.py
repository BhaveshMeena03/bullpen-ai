"""HTTP client for the support concierge (`POST /v1/chat`).

A sibling of client.py, which talks to podcast search. Kept separate because
the two endpoints answer different questions and return different shapes: the
podcast endpoint returns transcript hits with timestamps, this one returns a
grounded answer plus the documentation pages it was drawn from.

Non-streaming on purpose. The web page uses /v1/chat/stream to fill text in as
it arrives, but a Discord interaction can't be updated token by token: you
defer once and edit once, so the full-JSON endpoint is the right one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


class ChatError(Exception):
    """kind is one of: timeout, http, bad_response."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class ChatResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    refused: bool = False


class ChatClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._base = base_url.rstrip("/")
        # Injectable so tests can drive a MockTransport, same as SearchClient.
        self._http = client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "bullpen-concierge-discord/1.0"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def ask(
        self,
        question: str,
        *,
        history: list[dict] | None = None,
        brief: bool = True,
        retries: int = 1,
    ) -> ChatResult:
        """POST the question to /v1/chat. Retries once on a transient failure,
        matching SearchClient: the backend sleeps on free hosting, so the first
        call after an idle period can legitimately time out once.

        `history` is prior turns as [{"role": ..., "content": ...}]. The API is
        stateless and the client replays the conversation, same as the web page.
        """
        url = f"{self._base}/v1/chat"
        # Brief by default from Discord: a full answer takes ~10s and lands as
        # a wall of text in a chat window. Brief is ~4s and fits on screen.
        # Callers can turn it off when someone explicitly asks for depth.
        payload: dict = {"message": question, "brief": brief}
        if history:
            payload["history"] = history
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = await self._http.post(url, json=payload)
                if resp.status_code == 429:
                    raise ChatError(
                        "http", "The assistant is at capacity right now."
                    )
                resp.raise_for_status()
                return _parse(resp.json())
            except httpx.TimeoutException as exc:
                last = ChatError("timeout", "The assistant took too long to answer.")
                if attempt >= retries:
                    raise last from exc
            except ChatError:
                raise
            except httpx.HTTPError as exc:
                last = ChatError("http", "Couldn't reach the assistant.")
                if attempt >= retries:
                    raise last from exc
        raise last or ChatError("http", "Couldn't reach the assistant.")


def _parse(data: dict) -> ChatResult:
    if not isinstance(data, dict):
        raise ChatError("bad_response", "Unexpected response from the assistant.")
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise ChatError("bad_response", "Unexpected response from the assistant.")

    # Cite each documentation page once. The retriever returns several chunks
    # per page, so the raw list repeats titles; a reader wants the page names.
    titles: list[str] = []
    for chunk in data.get("sources") or []:
        if not isinstance(chunk, dict):
            continue
        meta = chunk.get("metadata") or {}
        title = meta.get("title") or meta.get("source_id") or chunk.get("id")
        if isinstance(title, str) and title not in titles:
            titles.append(title)

    return ChatResult(
        answer=answer,
        sources=titles,
        refused=bool(data.get("refused", False)),
    )
