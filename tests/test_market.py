"""Token resolution must refuse to guess.

A symbol lifted from an auto-generated caption is not an identity. These
tests pin the cases where returning None is the correct answer, because the
failure mode is a public swap link pointing at somebody else's token.
"""
import pytest

from app.market import (
    MIN_LIQUIDITY_USD,
    build_symbol_table,
    clean_symbol,
    from_coingecko,
    pick_token,
    summarise,
    valid_mint,
)

# Structurally valid Solana mints (strict base58, 32-44 chars).
MINT_A = "A" * 44
MINT_B = "B" * 44
MINT_C = "C" * 43


def tok(symbol="PENGU", *, mint=MINT_A, verified=True, liquidity=1_000_000, **kw):
    return {"id": mint, "symbol": symbol, "name": symbol,
            "isVerified": verified, "liquidity": liquidity, **kw}


# --- mint validation -------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "MINT1",                 # too short
    "",                      # empty
    None,                    # not a string
    123,                     # not a string
    "0" * 44,                # 0 is not in the base58 alphabet
    "O" * 44,                # nor O
    "I" * 44,                # nor I
    "l" * 44,                # nor l
    "A" * 45,                # too long
    "A" * 31,                # too short by one
    "../../etc/passwd",      # path traversal
    "https://evil.example",  # a URL, not a mint
    "A" * 43 + "/",          # trailing separator
])
def test_invalid_mints_are_rejected(bad):
    assert valid_mint(bad) is False


@pytest.mark.parametrize("good", [MINT_A, MINT_C,
                                  "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
                                  "So11111111111111111111111111111111111111112"])
def test_real_shaped_mints_are_accepted(good):
    assert valid_mint(good) is True


def test_token_with_unusable_mint_is_rejected_entirely():
    """Upstream returning junk must not become a published link."""
    assert pick_token([tok(mint="../../evil")], "PENGU") is None
    assert pick_token([tok(mint="https://evil.example")], "PENGU") is None


# --- symbol validation -----------------------------------------------------

@pytest.mark.parametrize("bad", [
    "", "   ", None, 42, "a" * 21, "-LEAD", ".LEAD",
    "PEN GU", "PENGU;DROP", "../PENGU", "PEN\nGU", "PEN%00GU", "<script>",
    "PEN\tGU", "PEN\x00GU", "PEN\rGU",
])
def test_hostile_symbols_are_refused(bad):
    assert clean_symbol(bad) is None


def test_symbols_are_normalised_upper():
    assert clean_symbol(" pengu ") == "PENGU"
    assert clean_symbol("w-eth.x") == "W-ETH.X"


def test_surrounding_whitespace_is_trimmed_not_refused():
    """A trailing newline is a typo. It is stripped, and cannot survive."""
    assert clean_symbol("PENGU\n") == "PENGU"
    assert clean_symbol("  PENGU\t") == "PENGU"


def test_no_control_character_survives_cleaning():
    for raw in ("PENGU\n", " PENGU ", "PENGU\t", "\rPENGU"):
        out = clean_symbol(raw)
        assert out is None or out.isalnum()


# --- resolution ------------------------------------------------------------

def test_picks_the_verified_match():
    assert pick_token([tok()], "PENGU")["id"] == MINT_A


def test_symbol_match_is_case_insensitive():
    assert pick_token([tok(symbol="Bonk")], "bonk") is not None


def test_a_leading_dollar_sign_is_not_part_of_the_identity():
    """The real WIF case, from production.

    Jupiter lists dogwifhat with the symbol "$WIF" while the extractor
    stores "WIF", so an exact compare rejected a verified token holding
    $5.6M of liquidity, and the dashboard showed no market data at all for
    one of the best known assets on Solana. assets.canonical() already
    strips the same character on the way in; the two sides were normalising
    differently.
    """
    assert pick_token([tok(symbol="$WIF")], "WIF") is not None
    assert pick_token([tok(symbol="WIF")], "$WIF") is not None


def test_stripping_the_dollar_sign_does_not_make_different_tickers_match():
    """The loosening above must not become fuzzy matching."""
    assert pick_token([tok(symbol="$WIF")], "WIFE") is None
    assert pick_token([tok(symbol="$BONK")], "BONKK") is None


def test_unverified_token_is_rejected():
    """The scam-collision case: right symbol, not verified."""
    assert pick_token([tok(verified=False)], "PENGU") is None


def test_verified_missing_is_rejected():
    t = tok()
    del t["isVerified"]
    assert pick_token([t], "PENGU") is None


def test_verified_truthy_but_not_true_is_rejected():
    """`"yes"` and `1` are not confirmation."""
    assert pick_token([tok(verified="yes")], "PENGU") is None
    assert pick_token([tok(verified=1)], "PENGU") is None


def test_thin_liquidity_is_rejected():
    assert pick_token([tok(liquidity=MIN_LIQUIDITY_USD - 1)], "PENGU") is None


def test_liquidity_exactly_at_threshold_is_accepted():
    assert pick_token([tok(liquidity=MIN_LIQUIDITY_USD)], "PENGU") is not None


def test_unparseable_liquidity_counts_as_zero_not_a_crash():
    for junk in ("lots", None, float("nan"), float("inf"), {}):
        assert pick_token([tok(liquidity=junk)], "PENGU") is None


def test_near_miss_symbol_is_not_fuzzy_matched():
    """PENGUIN must never resolve a request for PENGU."""
    assert pick_token([tok(symbol="PENGUIN")], "PENGU") is None


def test_deepest_liquidity_wins_among_verified_matches():
    shallow = tok(mint=MINT_A, liquidity=100_000)
    deep = tok(mint=MINT_B, liquidity=9_000_000)
    assert pick_token([shallow, deep], "PENGU")["id"] == MINT_B


def test_unverified_whale_loses_to_verified_minnow():
    """Liquidity never overrides verification."""
    scam = tok(mint=MINT_A, verified=False, liquidity=50_000_000)
    real = tok(mint=MINT_B, liquidity=60_000)
    assert pick_token([scam, real], "PENGU")["id"] == MINT_B


def test_empty_inputs_return_none():
    assert pick_token([], "PENGU") is None
    assert pick_token([tok()], "") is None
    assert pick_token([tok()], None) is None


# --- output ----------------------------------------------------------------

def test_summarise_builds_swap_url_from_the_resolved_mint():
    out = summarise(tok(mint=MINT_A, usdPrice=1.5,
                        stats24h={"priceChange": -4.2}))
    assert out["trade"]["mint"] == MINT_A
    assert out["trade"]["swap_url"] == f"https://jup.ag/swap/USDC-{MINT_A}"
    assert out["change_24h_pct"] == -4.2
    assert out["source"] == "jupiter"


def test_summarise_survives_missing_stats():
    assert summarise(tok())["change_24h_pct"] is None


def test_swap_url_host_is_always_jupiter():
    """No upstream field may redirect where the link points."""
    out = summarise(tok(mint=MINT_A, name="https://evil.example"))
    assert out["trade"]["swap_url"].startswith("https://jup.ag/swap/")


# --- price and tradeability are separate answers ---------------------------

def test_a_jupiter_hit_is_priced_and_tradeable():
    out = summarise(tok(mint=MINT_A, usdPrice=2.0))
    assert out["price_usd"] == 2.0
    assert out["trade"] is not None


def test_a_coingecko_row_is_priced_but_never_tradeable():
    """This is the whole point of the split: BTC gets a price, not a button."""
    out = from_coingecko({"symbol": "btc", "name": "Bitcoin",
                          "current_price": 77281,
                          "price_change_percentage_24h": 0.2})
    assert out["price_usd"] == 77281
    assert out["symbol"] == "BTC"
    assert out["source"] == "coingecko"
    assert out["trade"] is None


def test_coingecko_trade_key_is_present_and_null_not_absent():
    """An absent key gets skipped by callers; a null gets handled."""
    assert "trade" in from_coingecko({"symbol": "x", "current_price": 1})


def test_coingecko_missing_change_is_none_not_zero():
    """Absent 24h data must not render as a flat 0.0%."""
    assert from_coingecko({"symbol": "x", "current_price": 1})["change_24h_pct"] is None


# --- the CoinGecko symbol table --------------------------------------------

def cg(symbol, price=1.0, name=None, **kw):
    return {"symbol": symbol, "name": name or symbol,
            "current_price": price, **kw}


def test_symbol_table_keeps_the_first_sighting():
    """Rows arrive market-cap descending, so first seen is the largest."""
    t = build_symbol_table([cg("btc", 77000, "Bitcoin"),
                            cg("btc", 0.0001, "BitcoinImpostor")])
    assert t["BTC"]["name"] == "Bitcoin"


def test_symbol_table_uppercases_keys():
    assert "ETH" in build_symbol_table([cg("eth")])


def test_symbol_table_skips_rows_with_no_price():
    t = build_symbol_table([{"symbol": "ghost", "current_price": None},
                            cg("real")])
    assert "GHOST" not in t
    assert "REAL" in t


def test_symbol_table_skips_unusable_symbols():
    t = build_symbol_table([{"symbol": None, "current_price": 1},
                            {"symbol": "", "current_price": 1},
                            {"symbol": "   ", "current_price": 1},
                            {"symbol": 42, "current_price": 1}])
    assert t == {}


def test_symbol_table_handles_an_empty_listing():
    assert build_symbol_table([]) == {}


def test_symbol_table_trims_whitespace_in_symbols():
    assert "SOL" in build_symbol_table([cg(" sol ")])


# --- quote(): which source answers, and what it is allowed to grant --------
#
# The suite has no pytest-asyncio; the house style is asyncio.run() inside a
# sync test, so these follow it rather than adding a dependency.

import asyncio  # noqa: E402

from app import market as _market  # noqa: E402

TABLE = {"BTC": {"symbol": "btc", "name": "Bitcoin", "current_price": 77281,
                 "price_change_percentage_24h": 0.2}}


def _no_solana(monkeypatch):
    async def _none(symbol, **kw):
        return None
    monkeypatch.setattr(_market, "lookup", _none)


def _on_solana(monkeypatch):
    async def _hit(symbol, **kw):
        return summarise(tok(symbol=symbol, mint=MINT_A, usdPrice=9.0))
    monkeypatch.setattr(_market, "lookup", _hit)


def test_solana_hit_wins_and_is_tradeable(monkeypatch):
    _on_solana(monkeypatch)
    q = asyncio.run(_market.quote("SOL", coingecko_table=TABLE))
    assert q["source"] == "jupiter"
    assert q["trade"] is not None


def test_falls_back_to_coingecko_priced_not_tradeable(monkeypatch):
    _no_solana(monkeypatch)
    q = asyncio.run(_market.quote("BTC", coingecko_table=TABLE))
    assert q["source"] == "coingecko"
    assert q["price_usd"] == 77281
    assert q["trade"] is None


def test_unknown_everywhere_is_none(monkeypatch):
    _no_solana(monkeypatch)
    assert asyncio.run(_market.quote("NVDA", coingecko_table=TABLE)) is None


def test_missing_table_still_allows_a_solana_quote(monkeypatch):
    _on_solana(monkeypatch)
    q = asyncio.run(_market.quote("SOL", coingecko_table=None))
    assert q is not None and q["trade"] is not None


def test_missing_table_means_no_fallback_price(monkeypatch):
    _no_solana(monkeypatch)
    assert asyncio.run(_market.quote("BTC", coingecko_table=None)) is None


def test_hostile_ticker_never_reaches_either_source(monkeypatch):
    called = []

    async def _spy(symbol, **kw):
        called.append(symbol)
        return None
    monkeypatch.setattr(_market, "lookup", _spy)
    for bad in ("../etc", "'; DROP--", "", "a" * 40, "<script>"):
        assert asyncio.run(_market.quote(bad, coingecko_table=TABLE)) is None
    assert called == [], "a rejected ticker must not be sent upstream"


def test_lowercase_ticker_matches_the_table(monkeypatch):
    _no_solana(monkeypatch)
    q = asyncio.run(_market.quote("btc", coingecko_table=TABLE))
    assert q is not None and q["symbol"] == "BTC"


def test_tradeable_always_implies_priced(monkeypatch):
    """The invariant: never a trade button without a price beside it."""
    _on_solana(monkeypatch)
    q = asyncio.run(_market.quote("SOL", coingecko_table=TABLE))
    assert q["trade"] is not None
    assert isinstance(q["price_usd"], (int, float))


def test_partial_coingecko_failure_still_returns_a_table(monkeypatch):
    """Two pages up, two down: keep the 500 symbols we did get."""
    class Resp:
        def __init__(self, rows): self._rows = rows
        def raise_for_status(self): pass
        def json(self): return self._rows

    calls = {"n": 0}

    class Client:
        async def get(self, url, params=None):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("rate limited")
            return Resp([cg(f"S{calls['n']}")])
        async def aclose(self): pass

    table = asyncio.run(_market.fetch_coingecko_table(client=Client(), pages=4))
    assert set(table) == {"S1", "S2"}, "partial results must survive"
