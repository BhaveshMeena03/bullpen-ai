"""Asset extraction normalization — the layer that turns raw model output
into an honest table.

The pilot run proved why this is needed: auto-captions mangled "$ANSEM" into
two separate top-10 assets ("ANOM" and "ANTHEM"), Pudgy Penguins appeared as
three rows, and a private company (Anthropic) and a platform (Polymarket)
were reported as tradeable assets with junk symbols.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from extract_assets import _windows  # noqa: E402

from app.assets import (  # noqa: E402
    aggregate,
)
from app.assets import (
    canonical as _canonical,
)
from app.assets import (
    deep_link as _deep_link,
)
from app.assets import (
    is_asset as _is_asset,
)
from app.assets import (
    timestamp as _timestamp,
)


def _hit(symbol, name="", kind="analysis", conf="high", ep="e1", t=60.0):
    return {
        "symbol": symbol, "name": name, "asset_class": "crypto", "kind": kind,
        "start_seconds": t, "note": "said things", "confidence": conf,
        "episode_id": ep, "episode_title": "Ep", "url": "https://youtu.be/x",
    }


class TestCanonical:
    def test_maps_spelled_out_names_to_tickers(self):
        assert _canonical("SOLANA", "Solana") == "SOL"
        assert _canonical("bitcoin", "Bitcoin") == "BTC"

    def test_merges_caption_manglings_of_ansem(self):
        # Confirmed from real pilot output — both described the $ANSEM thesis.
        assert _canonical("ANOM", "Anom") == "ANSEM"
        assert _canonical("ANTHEM", "Anthem") == "ANSEM"
        assert _canonical("ANSEM", "Ansem") == "ANSEM"

    def test_merges_pudgy_penguin_variants(self):
        assert _canonical("PUDGY", "Pudgy Penguins") == "PENGU"
        assert _canonical("PENGUINS", "Pudgy Penguins") == "PENGU"
        assert _canonical("PENGU", "Pengu") == "PENGU"

    def test_falls_back_to_uppercase_stripped_symbol(self):
        assert _canonical("$wif", "dogwifhat") == "WIF"


class TestIsAsset:
    def test_rejects_junk_symbols(self):
        assert not _is_asset("N/A", "Polymarket")
        assert not _is_asset("UNKNOWN", "Anthropic")
        assert not _is_asset("", "something")

    def test_rejects_categories_and_non_tradeables(self):
        assert not _is_asset("NFT", "NFTs")
        assert not _is_asset("BTC", "Anthropic")      # blocked by name
        assert not _is_asset("XYZ", "Polymarket")     # platform, no token

    def test_rejects_overlong_symbols(self):
        # "PENGUINS" is a name the model failed to ticker-ise.
        assert not _is_asset("PENGUINS", "Pudgy Penguins")

    def test_accepts_real_tickers(self):
        assert _is_asset("SOL", "Solana")
        assert _is_asset("AAPL", "Apple")
        assert _is_asset("ANSEM", "Ansem")


class TestAggregate:
    def test_merges_mangled_variants_into_one_asset(self):
        out = aggregate([
            _hit("ANOM", "Anom"), _hit("ANTHEM", "Anthem"), _hit("ANSEM", "Ansem"),
        ])
        assert len(out["assets"]) == 1
        a = out["assets"][0]
        assert a["symbol"] == "ANSEM"
        assert a["mentions"] == 3
        assert a["name"] == "Ansem (The Black Bull)"   # canonical display name

    def test_drops_non_assets_and_counts_them(self):
        out = aggregate([
            _hit("SOL", "Solana"),
            _hit("UNKNOWN", "Anthropic"),
            _hit("N/A", "Polymarket"),
        ])
        assert [a["symbol"] for a in out["assets"]] == ["SOL"]
        assert out["dropped_not_an_asset"] == 2

    def test_confidence_floor_filters_low(self):
        out = aggregate(
            [_hit("SOL", "Solana", conf="low"), _hit("BTC", "Bitcoin", conf="high")],
            min_confidence="medium",
        )
        assert [a["symbol"] for a in out["assets"]] == ["BTC"]
        assert out["dropped_low_confidence"] == 1

    def test_ranks_by_analysis_depth_not_raw_mentions(self):
        hits = [_hit("SOL", "Solana", kind="analysis")]
        hits += [_hit("BTC", "Bitcoin", kind="mention") for _ in range(9)]
        out = aggregate(hits)
        # BTC has 9x the mentions but zero real analysis — SOL ranks first.
        assert out["assets"][0]["symbol"] == "SOL"

    def test_counts_distinct_episodes(self):
        out = aggregate([
            _hit("SOL", "Solana", ep="e1"), _hit("SOL", "Solana", ep="e1"),
            _hit("SOL", "Solana", ep="e2"),
        ])
        assert out["assets"][0]["episode_count"] == 2
        assert out["assets"][0]["mentions"] == 3


class TestWindows:
    def test_windows_keep_start_timestamp_and_respect_budget(self):
        segs = [{"t": float(i * 10), "text": "x" * 100} for i in range(10)]
        wins = _windows(segs, 300)
        assert len(wins) > 1
        assert wins[0][0] == 0.0
        assert wins[1][0] > 0.0            # later window starts later
        assert "[t=0]" in wins[0][1]       # timestamp markers survive

    def test_single_window_when_under_budget(self):
        segs = [{"t": 0.0, "text": "short"}]
        assert len(_windows(segs, 5000)) == 1


class TestFormatting:
    def test_timestamp_formats_hours_only_when_needed(self):
        assert _timestamp(65) == "1:05"
        assert _timestamp(3725) == "1:02:05"

    def test_deep_link_appends_time_param(self):
        assert _deep_link("https://youtu.be/x", 61) == "https://youtu.be/x?t=61s"
        assert _deep_link("https://y/w?v=1", 5) == "https://y/w?v=1&t=5s"
        assert _deep_link("", 5) == ""
