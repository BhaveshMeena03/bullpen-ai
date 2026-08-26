"""Rate limiting and admin auth."""

import asyncio

import pytest
from fastapi import HTTPException
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

    def __init__(self, *args, **kwargs):
        # The real classes take a usage ledger; ignore it here.
        pass

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

        # This originally asserted the LEFT-most hop, which is what the code
        # did and what made the limiter bypassable — that entry is whatever the
        # caller typed. The chains below are real ones read back from
        # production: client, then Cloudflare, then Render.
        assert limiter._client_ip(
            req("1.1.1.1, 172.69.179.154, 10.192.63.131")) == "1.1.1.1"
        assert limiter._client_ip(
            req("9.9.9.9, 1.1.1.1, 172.69.179.154, 10.192.63.131")) == "1.1.1.1"
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


class TestClientIpCannotBeChosenByTheCaller:
    """The per-IP limiter keyed on a caller-supplied value.

    Every chain below was copied from a real request to production via
    /v1/whoami, not invented:

        no spoof : 49.36.72.251, 172.69.179.154, 10.192.63.131
        spoofed  : 6.6.6.6, 49.36.72.251, 172.69.86.84, 10.192.63.131
                   ^forged   ^real client  ^cloudflare   ^render

    Cloudflare prepends nothing and appends nothing: it passes the caller's
    header through and the real address lands immediately after it.
    """

    REAL = "49.36.72.251"
    CF = "172.69.179.154"
    RENDER = "10.192.63.131"

    @staticmethod
    def _req(xff=None, client_host="10.192.63.131", **headers):
        from starlette.requests import Request
        raw = [(k.replace("_", "-").encode(), v.encode())
               for k, v in headers.items()]
        if xff:
            raw.append((b"x-forwarded-for", xff.encode()))
        return Request({"type": "http", "headers": raw, "method": "GET",
                        "path": "/", "scheme": "https",
                        "client": (client_host, 1234)})

    def test_genuine_chain_resolves_to_the_real_client(self):
        from app.security import RateLimiter
        assert RateLimiter._client_ip(
            self._req(f"{self.REAL}, {self.CF}, {self.RENDER}")) == self.REAL

    def test_forged_leading_entry_is_ignored(self):
        from app.security import RateLimiter
        assert RateLimiter._client_ip(
            self._req(f"6.6.6.6, {self.REAL}, {self.CF}, {self.RENDER}")) == self.REAL

    def test_cloudflare_header_wins_and_is_not_forgeable_from_outside(self):
        from app.security import RateLimiter
        # Cloudflare 403s a request carrying its own CF-Connecting-IP, so a
        # value reaching us here was written by Cloudflare.
        assert RateLimiter._client_ip(
            self._req(f"6.6.6.6, {self.REAL}, {self.CF}, {self.RENDER}",
                      cf_connecting_ip=self.REAL)) == self.REAL

    def test_rotating_cloudflare_edge_does_not_open_new_buckets(self):
        """The regression that switched per-IP limiting off entirely.

        Cloudflare edge addresses are public and rotate, so 'right-most public
        address' handed every request a fresh bucket.
        """
        from app.security import RateLimiter
        edges = ["172.69.179.154", "172.68.175.47", "172.69.86.84"]
        seen = {RateLimiter._client_ip(
            self._req(f"{self.REAL}, {e}, {self.RENDER}")) for e in edges}
        assert seen == {self.REAL}, f"bucket key moved with the edge: {seen}"

    def test_rotating_forgeries_collapse_to_one_bucket(self):
        from app.security import RateLimiter
        seen = {RateLimiter._client_ip(
            self._req(f"203.0.113.{i}, {self.REAL}, {self.CF}, {self.RENDER}"))
            for i in range(1, 30)}
        assert seen == {self.REAL}

    def test_rotating_forgeries_actually_get_throttled(self):
        """End to end through the bucket, not just the parsing."""
        from app.security import RateLimiter
        limiter = RateLimiter(rpm=30, burst=5)
        allowed = sum(
            limiter.check(RateLimiter._client_ip(self._req(
                f"203.0.113.{i}, {self.REAL}, {self.CF}, {self.RENDER}")))
            for i in range(1, 40))
        assert allowed <= 6, f"{allowed} of 39 forged requests allowed"

    def test_direct_request_without_proxy_headers(self):
        from app.security import RateLimiter
        assert RateLimiter._client_ip(
            self._req(client_host="10.0.0.1")) == "10.0.0.1"

    def test_short_chain_falls_back_to_the_peer(self):
        """A truncated chain must never index into the caller's own entry."""
        from app.security import RateLimiter
        assert RateLimiter._client_ip(
            self._req("6.6.6.6", client_host="10.0.0.9")) == "10.0.0.9"

    def test_private_value_in_trusted_header_is_not_used(self):
        from app.security import RateLimiter
        assert RateLimiter._client_ip(
            self._req(f"{self.REAL}, {self.CF}, {self.RENDER}",
                      cf_connecting_ip="10.0.0.5")) == self.REAL


class TestPerClientDailyBudget:
    """One client must not be able to drain the whole day's budget.

    The per-minute limiter caps rate, not total: a script sitting exactly on
    12/min reached the 3000/day ceiling in about four hours, and every request
    looked legal on the way there. After that, genuine visitors were refused
    until UTC midnight.
    """

    def _req(self, ip: str):
        from starlette.requests import Request

        return Request({"type": "http", "headers": [], "client": (ip, 1234),
                        "method": "GET", "path": "/", "scheme": "https"})

    def test_one_client_is_capped(self):
        from app.security import PerClientDailyBudget

        budget = PerClientDailyBudget(limit=5)
        for _ in range(5):
            asyncio.run(budget(self._req("1.2.3.4")))
        with pytest.raises(HTTPException) as exc:
            asyncio.run(budget(self._req("1.2.3.4")))
        assert exc.value.status_code == 429
        assert "midnight" in exc.value.detail.lower()

    def test_one_client_hitting_the_cap_does_not_block_others(self):
        """The whole point: a drained attacker must not deny everyone else."""
        from app.security import PerClientDailyBudget

        budget = PerClientDailyBudget(limit=3)
        for _ in range(3):
            asyncio.run(budget(self._req("9.9.9.9")))
        with pytest.raises(HTTPException):
            asyncio.run(budget(self._req("9.9.9.9")))
        asyncio.run(budget(self._req("5.5.5.5")))   # unaffected

    def test_zero_disables(self):
        from app.security import PerClientDailyBudget

        budget = PerClientDailyBudget(limit=0)
        for _ in range(50):
            asyncio.run(budget(self._req("1.2.3.4")))

    def test_client_table_is_bounded(self):
        """Memory must not grow with attacker-controlled IP churn."""
        from app.security import PerClientDailyBudget

        budget = PerClientDailyBudget(limit=5)
        budget.MAX_CLIENTS = 50
        for i in range(400):
            asyncio.run(budget(self._req(f"10.0.{i // 256}.{i % 256}")))
        assert len(budget._counts) <= 50


class TestCachedAnswersAreFree:
    """A cache hit costs nothing, so it must not spend the caller's day.

    Found the way these things usually are: the person who owns the site
    could not use his own demo. A verification run had spent his allowance,
    and most of what it spent went on answers that were served from cache
    for $0. The limiter is there to bound spend, so charging for a free
    response made the cache cheaper without making the service any more
    available — which was half the reason for building it.
    """

    def _req(self, ip: str):
        from starlette.requests import Request

        return Request({"type": "http", "headers": [], "client": (ip, 1234),
                        "method": "GET", "path": "/", "scheme": "https"})

    def test_a_refunded_request_does_not_count(self):
        from app.security import PerClientDailyBudget

        budget = PerClientDailyBudget(limit=3)
        for _ in range(10):
            asyncio.run(budget(self._req("1.2.3.4")))   # charged on the way in
            budget.refund(self._req("1.2.3.4"))         # ...cache had it
        asyncio.run(budget(self._req("1.2.3.4")))       # still has room

    def test_refunds_are_per_client(self):
        """One client's cache hit must not top up another's allowance."""
        from app.security import PerClientDailyBudget

        budget = PerClientDailyBudget(limit=2)
        asyncio.run(budget(self._req("1.1.1.1")))
        asyncio.run(budget(self._req("2.2.2.2")))
        budget.refund(self._req("1.1.1.1"))
        asyncio.run(budget(self._req("2.2.2.2")))       # 2.2.2.2 now at 2
        with pytest.raises(HTTPException):
            asyncio.run(budget(self._req("2.2.2.2")))

    def test_refund_never_goes_negative(self):
        """A refund with nothing charged must not hand out free credit."""
        from app.security import PerClientDailyBudget

        budget = PerClientDailyBudget(limit=1)
        for _ in range(5):
            budget.refund(self._req("1.2.3.4"))
        asyncio.run(budget(self._req("1.2.3.4")))
        with pytest.raises(HTTPException):
            asyncio.run(budget(self._req("1.2.3.4")))

    def test_refund_is_a_noop_when_disabled(self):
        from app.security import PerClientDailyBudget

        budget = PerClientDailyBudget(limit=0)
        budget.refund(self._req("1.2.3.4"))     # must not raise


def test_a_cached_search_does_not_spend_the_callers_allowance(monkeypatch):
    """The wiring, not just the method.

    RateLimiter.refund can be perfect and the bug still be live if a handler
    forgets to call it — which is exactly the shape of the original problem,
    since the limiter runs as a dependency and cannot see whether the cache
    had the answer. So this goes through the real endpoint twice: the first
    call is charged, the second is served from cache and must give the slot
    back.
    """
    from app.answer_cache import AnswerCache
    from app.schemas import PodcastSearchResponse
    from app.security import per_client_daily

    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    get_settings.cache_clear()

    answer = PodcastSearchResponse(answer="he said it here", hits=[],
                                   model="claude-haiku-4-5")

    class _Index(_Stub):
        calls = 0

        async def search(self, *a, **k):
            _Index.calls += 1
            return answer

    monkeypatch.setattr(main_module, "Retriever", _Stub)
    monkeypatch.setattr(main_module, "ConciergeAgent", _Stub)
    monkeypatch.setattr(main_module, "IngestionPipeline", _Stub)
    monkeypatch.setattr(main_module, "PodcastIndex", _Index)

    with TestClient(main_module.app) as client:
        # One shared instance: a fresh AnswerCache per request would never
        # hit, and the test would pass for the wrong reason.
        shared = AnswerCache()
        main_module.app.dependency_overrides[main_module.get_answers] = \
            lambda: shared
        try:
            per_client_daily._counts.clear()
            body = {"query": "what did ansem say about eth"}
            assert client.post("/v1/podcast/search", json=body).status_code == 200
            charged = dict(per_client_daily._counts)
            assert client.post("/v1/podcast/search", json=body).status_code == 200
            assert _Index.calls == 1, "second call should have been cached"
            assert per_client_daily._counts == charged, (
                "the cached answer cost nothing and must not have been "
                "charged against the caller's daily allowance")
        finally:
            main_module.app.dependency_overrides.clear()
            per_client_daily._counts.clear()
    get_settings.cache_clear()


def test_the_usage_report_is_not_public(admin_client):
    """What the service costs to run, how many calls each surface takes and
    which models answer them is operational detail about a business. Nothing
    in it is a credential, which is why it was public; that is not the same
    as it being nobody else's business."""
    assert admin_client.get("/v1/usage").status_code in (401, 403)
    ok = admin_client.get("/v1/usage", headers={"X-Admin-Token": "s3cret"})
    assert ok.status_code == 200
    assert "days" in ok.json()


def test_every_endpoint_that_reveals_operations_is_gated(admin_client):
    """A checklist, so a new one does not quietly ship open."""
    for path in ("/v1/usage", "/v1/whoami", "/v1/gaps"):
        assert admin_client.get(path).status_code in (401, 403), \
            f"{path} answers without a token"
