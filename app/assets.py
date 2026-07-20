"""Asset normalization and aggregation.

Pure logic, no I/O — shared by the extraction script (which produces raw
hits) and the API (which serves the aggregate). Keeping it here means the
live endpoint and the offline script can never drift apart.

The normalization exists because auto-generated captions mis-hear ticker
names. The pilot run produced "$ANSEM" as two separate top-10 assets
("ANOM" and "ANTHEM"), Pudgy Penguins as three rows, and reported a private
company (Anthropic) and a platform (Polymarket) as tradeable assets.
"""

from __future__ import annotations

# Canonical symbols. Several entries merge on NAME rather than symbol, which
# is far more reliable when the caption garbled the ticker itself.
ALIASES = {
    "solana": "SOL", "sol": "SOL", "soul": "SOL",
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH", "ether": "ETH",
    "hyperliquid": "HYPE", "hype": "HYPE",
    "bonk": "BONK", "bunk": "BONK",
    "dogecoin": "DOGE", "doge": "DOGE",
    "ansem": "ANSEM", "anom": "ANSEM", "anthem": "ANSEM",
    "pudgy": "PENGU", "penguins": "PENGU", "pudgy penguins": "PENGU",
    "pengu": "PENGU",
    "usdc": "USDC", "tether": "USDT", "usdt": "USDT",
    "ripple": "XRP", "xrp": "XRP",
    "toncoin": "TON", "telegram": "TON",
    "apple": "AAPL", "aapl": "AAPL",
    "nvidia": "NVDA", "nvda": "NVDA",
    "tesla": "TSLA", "tsla": "TSLA",
    "microstrategy": "MSTR", "mstr": "MSTR",
    "robinhood": "HOOD", "coinbase": "COIN", "blackrock": "BLK",
    "gamestop": "GME", "zcash": "ZEC", "binance coin": "BNB",
    "s&p 500": "SPX", "pump fun": "PUMP", "pump.fun": "PUMP",
    "coreweave": "CRWV",
    "cerebras": "CBRS", "cerebrus": "CBRS",
    "sandisk": "SNDK", "sandis": "SNDK",
    "solstice": "SOLS", "solstice advanced materials": "SOLS",
    "take-two interactive": "TTWO", "take two interactive": "TTWO",
    "venice": "VVV", "venice ai": "VVV",
}

# Canonical display names, so merged variants show one consistent label.
NAMES = {
    "ANSEM": "Ansem (The Black Bull)", "PENGU": "Pudgy Penguins",
    "TON": "Toncoin", "SPX": "S&P 500", "HYPE": "Hyperliquid",
    "PUMP": "Pump.fun", "SOL": "Solana", "BTC": "Bitcoin", "ETH": "Ethereum",
}

BLOCK_SYMBOLS = {
    "N/A", "NA", "UNKNOWN", "NONE", "NFT", "NFTS", "PUNK", "",
    "AI", "CRYPTO",  # emitted as literal "(general)" categories
}
BLOCK_NAMES = {
    "anthropic", "openai", "chatgpt", "gpt", "polymarket", "kraken",
    "nfts", "nft", "non-fungible tokens", "cryptopunks", "twitter", "x",
    "spacex", "xai",                       # private companies, not tickers
    "artificial intelligence (general)", "crypto (general)",
    "artificial intelligence", "stocks", "the market",
}

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def canonical(symbol: str, name: str) -> str:
    for key in (symbol.strip().lower(), name.strip().lower()):
        if key in ALIASES:
            return ALIASES[key]
    return symbol.strip().upper().lstrip("$")


def is_asset(symbol: str, name: str) -> bool:
    """Filter out things that aren't tradeable assets. The model correctly
    reports what was discussed, but 'Anthropic', 'Polymarket' and 'NFTs' are
    a private company, a platform, and a category — not tickers."""
    if symbol in BLOCK_SYMBOLS or name.strip().lower() in BLOCK_NAMES:
        return False
    # A "symbol" longer than 6 chars is almost always a name the model
    # couldn't ticker-ise (e.g. "PENGUINS"), unless aliased explicitly.
    return bool(symbol) and symbol.isalnum() and len(symbol) <= 6


def timestamp(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def deep_link(url: str, seconds: float) -> str:
    if not url:
        return ""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={int(seconds)}s"


def aggregate(hits: list[dict], min_confidence: str = "medium") -> dict:
    """Roll per-window hits up into one record per asset."""
    floor = CONFIDENCE_RANK[min_confidence]

    kept, blocked = [], 0
    for h in hits:
        if CONFIDENCE_RANK.get(h.get("confidence", "low"), 0) < floor:
            continue
        h = dict(h)
        h["symbol"] = canonical(h["symbol"], h.get("name", ""))
        if not is_asset(h["symbol"], h.get("name", "")):
            blocked += 1
            continue
        kept.append(h)

    assets: dict[str, dict] = {}
    for h in kept:
        a = assets.setdefault(h["symbol"], {
            "symbol": h["symbol"],
            "name": NAMES.get(h["symbol"], h.get("name", "")),
            "asset_class": h.get("asset_class", "other"),
            "mentions": 0, "analysis": 0, "episodes": set(), "moments": [],
        })
        a["mentions"] += 1
        if h.get("kind") == "analysis":
            a["analysis"] += 1
        if not a["name"] and h.get("name"):
            a["name"] = h["name"]
        a["episodes"].add(h.get("episode_id", ""))
        a["moments"].append({
            "episode_id": h.get("episode_id", ""),
            "episode_title": h.get("episode_title", ""),
            "start_seconds": h.get("start_seconds", 0),
            "timestamp": timestamp(h.get("start_seconds", 0)),
            "deep_link": deep_link(h.get("url", ""), h.get("start_seconds", 0)),
            "kind": h.get("kind", "mention"),
            "note": h.get("note", ""),
        })

    out = []
    for a in assets.values():
        a["moments"].sort(key=lambda m: (m["episode_id"], m["start_seconds"]))
        a["episode_count"] = len(a["episodes"])
        del a["episodes"]
        # A single reference across every episode is where caption errors
        # live. Flag it so the UI can keep it out of the headline table
        # without hiding the data entirely.
        a["single_reference"] = a["mentions"] < 2
        out.append(a)
    # Rank by depth of discussion first, then raw mentions.
    out.sort(key=lambda a: (a["analysis"], a["mentions"]), reverse=True)
    return {
        "assets": out,
        "total_hits": len(kept),
        "dropped_low_confidence": sum(
            1 for h in hits
            if CONFIDENCE_RANK.get(h.get("confidence", "low"), 0) < floor
        ),
        "dropped_not_an_asset": blocked,
    }
