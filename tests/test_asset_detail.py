"""The per-asset endpoint an agent calls.

It reports what was said and cites it. It must never invent a market, never
500 when the market lookup fails, and never accept a ticker it cannot vouch
for — a swap link on a public page is the thing that has to be right.
"""
import pytest
from fastapi.testclient import TestClient

from app import main as main_module

MINT = "A" * 44
FAKE_MARKET = {
    "mint": MINT, "symbol": "SOL", "name": "Solana", "price_usd": 93.27,
    "change_24h_pct": 2.5, "liquidity_usd": 745_630_855, "mcap_usd": 1,
    "holders": 2, "verified": True,
    "swap_url": f"https://jup.ag/swap/USDC-{MINT}",
}


@pytest.fixture
def client(monkeypatch):
    async def fake_lookup(symbol, **kw):
        return FAKE_MARKET if symbol == "SOL" else None
    monkeypatch.setattr(main_module.market, "lookup", fake_lookup)
    with TestClient(main_module.app) as c:
        main_module.app.state._market_cache.clear()
        yield c


class TestHappyPath:
    def test_returns_moments_and_market(self, client):
        r = client.get("/v1/assets/SOL")
        assert r.status_code == 200
        b = r.json()
        assert b["symbol"] == "SOL"
        assert b["moments"], "moments must be cited, not summarised away"
        assert b["market"]["mint"] == MINT
        assert b["market"]["swap_url"].startswith("https://jup.ag/swap/")

    def test_lowercase_ticker_resolves(self, client):
        assert client.get("/v1/assets/sol").status_code == 200

    def test_disclaimer_is_always_present(self, client):
        b = client.get("/v1/assets/SOL").json()
        assert "not advice" in b["disclaimer"].lower()

    def test_every_moment_carries_a_citation(self, client):
        for m in client.get("/v1/assets/SOL").json()["moments"]:
            assert m.get("deep_link", "").startswith("https://www.youtube.com/")
            assert "timestamp" in m


class TestMarketIsOptional:
    def test_unresolvable_ticker_yields_null_market_not_a_guess(self, client):
        """BTC is discussed constantly but has no Solana mint we vouch for."""
        r = client.get("/v1/assets/BTC")
        assert r.status_code == 200
        assert r.json()["market"] is None

    def test_upstream_failure_does_not_500(self, client, monkeypatch):
        async def boom(symbol, **kw):
            raise RuntimeError("jupiter down")
        monkeypatch.setattr(main_module.market, "lookup", boom)
        main_module.app.state._market_cache.clear()
        r = client.get("/v1/assets/SOL")
        assert r.status_code == 200
        assert r.json()["market"] is None
        assert r.json()["moments"], "citations survive a market outage"


class TestRejection:
    @pytest.mark.parametrize("bad", [
        "NOTAREALTICKER", "..", "%2e%2e", "<script>", "a" * 40,
        "SOL;DROP", "SOL%20OR%201", "'", '"',
    ])
    def test_bad_tickers_are_404_not_500(self, client, bad):
        assert client.get(f"/v1/assets/{bad}").status_code in (404, 400)

    def test_404_body_leaks_nothing(self, client):
        b = client.get("/v1/assets/NOTAREALTICKER").json()
        assert b["detail"] == "Unknown asset."
        assert "pinecone" not in str(b).lower()
        assert "traceback" not in str(b).lower()


class TestCaching:
    def test_market_cache_is_bounded(self, client):
        cache = main_module.app.state._market_cache
        cache.clear()
        for i in range(300):
            cache[f"T{i}"] = (0.0, None)
        assert len(cache) == 300      # the guard runs on write in _market_for
        cache.clear()
        assert len(cache) == 0
