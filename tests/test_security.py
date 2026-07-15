"""Rate limiting and admin auth."""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import get_settings
from app.security import RateLimiter


class TestRateLimiter:
    def test_allows_within_burst(self):
        limiter = RateLimiter(rpm=60, burst=5)
        assert all(limiter.check("1.2.3.4") for _ in range(5))

    def test_blocks_after_burst(self):
        limiter = RateLimiter(rpm=60, burst=3)
        for _ in range(3):
            limiter.check("1.2.3.4")
        assert limiter.check("1.2.3.4") is False

    def test_keys_are_independent(self):
        limiter = RateLimiter(rpm=60, burst=1)
        assert limiter.check("a")
        assert limiter.check("b"), "one client's burst must not affect another"

    def test_refills_over_time(self, monkeypatch):
        limiter = RateLimiter(rpm=60, burst=1)  # 1 token/sec
        clock = [1000.0]
        monkeypatch.setattr("app.security.time.monotonic", lambda: clock[0])
        assert limiter.check("x")
        assert limiter.check("x") is False
        clock[0] += 2.0  # two seconds pass -> bucket refills
        assert limiter.check("x")


class _Stub:
    """Minimal stand-ins so TestClient can start the app without real keys."""

    async def search(self, *a, **k):
        return []

    async def answer(self, *a, **k):  # pragma: no cover - not exercised here
        raise AssertionError("not used")

    async def ingest(self, docs):
        return len(docs)


@pytest.fixture
def admin_client(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    get_settings.cache_clear()
    monkeypatch.setattr(main_module, "Retriever", _Stub)
    monkeypatch.setattr(main_module, "ConciergeAgent", _Stub)
    monkeypatch.setattr(main_module, "IngestionPipeline", _Stub)
    monkeypatch.setattr(main_module, "PodcastIndex", _Stub)
    with TestClient(main_module.app) as client:
        yield client
    get_settings.cache_clear()


DOC = [{"source_type": "docs", "source_id": "d", "text": "hello"}]


class TestAdminAuth:
    def test_ingest_rejects_missing_token(self, admin_client):
        assert admin_client.post("/v1/ingest", json=DOC).status_code == 401

    def test_ingest_rejects_wrong_token(self, admin_client):
        r = admin_client.post(
            "/v1/ingest", json=DOC, headers={"X-Admin-Token": "nope"}
        )
        assert r.status_code == 401

    def test_ingest_accepts_correct_token(self, admin_client):
        r = admin_client.post(
            "/v1/ingest", json=DOC, headers={"X-Admin-Token": "s3cret"}
        )
        assert r.status_code == 200

    def test_podcast_ingest_also_guarded(self, admin_client):
        r = admin_client.post("/v1/podcast/ingest", json=[])
        assert r.status_code == 401
