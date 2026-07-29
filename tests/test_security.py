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


class TestSecretStripping:
    def test_keys_are_stripped(self, monkeypatch):
        # Trailing newline, zero-width space (U+200B), and NBSP (U+00A0) —
        # all observed corruptions from real dashboard pastes.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-abc\n")
        monkeypatch.setenv("VOYAGE_API_KEY", "pa-x\u200byz\n")
        monkeypatch.setenv("PINECONE_API_KEY", "\u00a0pcsk-123 ")
        get_settings.cache_clear()
        s = get_settings()
        assert s.anthropic_api_key == "sk-ant-abc"
        assert s.voyage_api_key == "pa-xyz"
        assert s.pinecone_api_key == "pcsk-123"
        get_settings.cache_clear()


class TestProxyIPAndBuckets:
    def test_forwarded_for_is_used(self):
        from starlette.requests import Request as SReq
        limiter = RateLimiter(rpm=60, burst=1)

        def req(xff):
            scope = {"type": "http", "headers": [(b"x-forwarded-for", xff.encode())],
                     "client": ("10.0.0.1", 0)}
            return SReq(scope)

        assert limiter._client_ip(req("1.1.1.1, 10.0.0.1")) == "1.1.1.1"
        # Two different real clients behind the same proxy get separate buckets
        assert limiter.check("1.1.1.1")
        assert limiter.check("2.2.2.2"), "distinct XFF clients must not share a bucket"

    def test_bucket_eviction_preserves_recent_throttled(self, monkeypatch):
        # Deterministic, monotonically-increasing clock so "oldest" is
        # unambiguous.
        # Tiny increment: preserves ordering (oldest < newest) without
        # refilling a meaningful fraction of a token between calls.
        clock = [1000.0]
        monkeypatch.setattr("app.security.time.monotonic",
                            lambda: clock.__setitem__(0, clock[0] + 0.001) or clock[0])
        limiter = RateLimiter(rpm=60, burst=1)
        limiter.MAX_BUCKETS = 2
        limiter.check("old")               # oldest bucket
        limiter.check("victim")            # newer; consumes its 1 token
        assert limiter.check("victim") is False   # victim now throttled
        limiter.check("newcomer")          # at cap -> evicts the OLDEST ("old")
        # victim (more recent) survived and is still throttled; a full clear()
        # would have wiped it and let it burst again.
        assert limiter.check("victim") is False


class TestAdminGuardWithNoTokenConfigured:
    """The blank-token case was covered; the *absent* case was not, and that
    was the one that mattered. With no ADMIN_TOKEN the guard used to return
    early and wave every caller through, so an environment that lost the
    variable published /v1/ingest and /v1/podcast/ingest to the internet with
    nothing in the response to say so.
    """

    @staticmethod
    def _request(host: str):
        from types import SimpleNamespace
        return SimpleNamespace(client=SimpleNamespace(host=host), headers={})

    def _call(self, host: str):
        import asyncio

        from app.security import require_admin
        return asyncio.run(require_admin(self._request(host)))

    def test_remote_caller_is_refused(self, monkeypatch):
        import pytest
        from fastapi import HTTPException
        monkeypatch.delenv("ADMIN_TOKEN", raising=False)
        get_settings.cache_clear()
        with pytest.raises(HTTPException) as exc:
            self._call("203.0.113.7")
        assert exc.value.status_code == 503
        get_settings.cache_clear()

    def test_unparseable_client_host_is_refused(self, monkeypatch):
        """Anything that isn't a loopback IP fails closed, including the
        placeholder hosts test clients and some proxies present."""
        import pytest
        from fastapi import HTTPException
        monkeypatch.delenv("ADMIN_TOKEN", raising=False)
        get_settings.cache_clear()
        with pytest.raises(HTTPException):
            self._call("testclient")
        get_settings.cache_clear()

    def test_loopback_still_works_for_local_dev(self, monkeypatch):
        monkeypatch.delenv("ADMIN_TOKEN", raising=False)
        get_settings.cache_clear()
        assert self._call("127.0.0.1") is None
        assert self._call("::1") is None
        get_settings.cache_clear()


class TestAdminTokenFailClosed:
    def test_blank_admin_token_refuses_to_start(self, monkeypatch):
        import pytest
        from pydantic import ValidationError
        monkeypatch.setenv("ADMIN_TOKEN", "​ \n")  # invisible-only
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()
        get_settings.cache_clear()


class TestHistorySpendCeiling:
    """Input is billed per token, so an unbounded replayed conversation makes
    a caller's spend a function of what they choose to send rather than of the
    rate limit. The per-field limits alone allowed 40 x 16,000 = 640,000
    characters, roughly 160k tokens, on every single request.
    """

    @staticmethod
    def _turns(n: int, size: int):
        from app.schemas import ChatTurn
        return [ChatTurn(role="user", content="x" * size) for _ in range(n)]

    def test_realistic_conversation_is_accepted(self):
        from app.schemas import ChatRequest
        # The web page replays at most 20 entries and an answer is ~2,000
        # characters, so this is the upper end of something genuine.
        req = ChatRequest(message="hi", history=self._turns(20, 2_000))
        assert len(req.history) == 20

    def test_previously_allowed_worst_case_is_rejected(self):
        import pytest
        from pydantic import ValidationError

        from app.schemas import ChatRequest
        with pytest.raises(ValidationError):
            ChatRequest(message="hi", history=self._turns(40, 16_000))

    def test_limit_counts_the_total_not_each_turn(self):
        """Many small turns that individually pass must still be capped."""
        import pytest
        from pydantic import ValidationError

        from app.schemas import MAX_HISTORY_CHARS, ChatRequest
        over = (MAX_HISTORY_CHARS // 1_000) + 2
        with pytest.raises(ValidationError):
            ChatRequest(message="hi", history=self._turns(over, 1_000))


class TestDailyBudget:
    """Per-minute limits stop a burst but not a slow drain: 25 requests a
    minute sits inside every other limit and still reaches ~36,000 in a day.
    """

    def test_allows_up_to_the_ceiling_then_refuses(self):
        from app.security import DailyBudget
        b = DailyBudget(limit=3)
        assert [b.check() for _ in range(3)] == [True, True, True]
        assert b.check() is False

    def test_resets_on_utc_rollover(self, monkeypatch):
        from app.security import DailyBudget
        b = DailyBudget(limit=1)
        monkeypatch.setattr(DailyBudget, "_today", staticmethod(lambda: "2026-07-29"))
        assert b.check() is True
        assert b.check() is False
        monkeypatch.setattr(DailyBudget, "_today", staticmethod(lambda: "2026-07-30"))
        assert b.check() is True, "a new UTC day must start fresh"

    def test_zero_disables_the_cap(self):
        from app.security import DailyBudget
        b = DailyBudget(limit=0)
        assert all(b.check() for _ in range(500))

    def test_exhausted_budget_returns_503_not_500(self):
        import asyncio

        import pytest
        from fastapi import HTTPException

        from app.security import DailyBudget
        b = DailyBudget(limit=1)
        b.check()
        with pytest.raises(HTTPException) as exc:
            asyncio.run(b(None))
        assert exc.value.status_code == 503
        assert "Retry-After" in exc.value.headers

    def test_state_reports_usage(self):
        from app.security import DailyBudget
        b = DailyBudget(limit=10)
        b.check()
        b.check()
        assert b.state()["used"] == 2
        assert b.state()["limit"] == 10
