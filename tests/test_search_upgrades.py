"""Search upgrades: reranking fallback, streaming endpoint, stats."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.embeddings import rerank_order
from app.schemas import PodcastHit


class TestRerankOrder:
    class _GoodVoyage:
        async def rerank(self, query, documents, model, top_k):
            class Res:  # best-first: reverse of input order
                results = [
                    type("R", (), {"index": i})()
                    for i in reversed(range(min(top_k, len(documents))))
                ]
            return Res()

    class _BrokenVoyage:
        async def rerank(self, *a, **k):
            raise RuntimeError("rate limited")

    def test_returns_reranked_indices(self):
        order = asyncio.run(rerank_order(
            self._GoodVoyage(), "q", ["a", "b", "c"], top_k=2, model="m"
        ))
        assert order == [1, 0]

    def test_failure_returns_none_not_raise(self):
        order = asyncio.run(rerank_order(
            self._BrokenVoyage(), "q", ["a", "b"], top_k=2, model="m"
        ))
        assert order is None, "rerank must degrade, never break search"


HIT = PodcastHit(
    episode_id="ep1", title="Test Ep", start_seconds=61,
    timestamp="1:01", deep_link="https://youtube.com/watch?v=x&t=61s",
    text="the transcript moment", score=0.9,
)


class StubPodcast:
    mode = "ok"

    async def retrieve(self, query, top_k=None):
        return [HIT]

    async def search(self, query, top_k=None):  # pragma: no cover
        raise AssertionError("stream endpoint must not call search()")

    async def answer_stream(self, query, hits):
        if self.mode == "refusal":
            yield "\x00REFUSAL\x00"
            return
        for token in ("Grounded", " answer"):
            yield token

    async def list_all(self):  # summaries stub-compat
        return []


class _Stub:
    async def search(self, *a, **k):
        return []

    async def ingest(self, docs):
        return len(docs)

    async def list_all(self):
        return []


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "Retriever", _Stub)
    monkeypatch.setattr(main_module, "ConciergeAgent", _Stub)
    monkeypatch.setattr(main_module, "IngestionPipeline", _Stub)
    monkeypatch.setattr(main_module, "PodcastIndex", StubPodcast)
    monkeypatch.setattr(main_module, "SummaryStore", _Stub)
    StubPodcast.mode = "ok"
    with TestClient(main_module.app) as c:
        yield c


class TestPodcastSearchStream:
    def test_hits_first_then_deltas_then_done(self, client):
        r = client.post("/v1/podcast/search/stream", json={"query": "hi"})
        assert r.status_code == 200
        frames = [f for f in r.text.split("\n\n") if f.strip()]
        assert frames[0].startswith("event: hits")
        hit_payload = json.loads(frames[0].split("data: ", 1)[1])
        assert hit_payload[0]["timestamp"] == "1:01"
        assert 'data: {"text": "Grounded"}' in frames[1]
        assert frames[-1].startswith("event: done")

    def test_refusal_event(self, client):
        StubPodcast.mode = "refusal"
        r = client.post("/v1/podcast/search/stream", json={"query": "hi"})
        assert "event: refusal" in r.text
        assert "\x00" not in r.text


class TestStats:
    def test_stats_counts_searches(self, client):
        before = client.get("/v1/stats").json()["podcast_searches"]
        client.post("/v1/podcast/search/stream", json={"query": "hi"})
        after = client.get("/v1/stats").json()["podcast_searches"]
        assert after == before + 1

    def test_stats_shape(self, client):
        s = client.get("/v1/stats").json()
        for key in ("started_at", "podcast_searches", "concierge_chats"):
            assert key in s
