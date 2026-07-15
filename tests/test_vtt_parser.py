"""VTT caption parser — the tricky bit of episode ingestion. YouTube's
auto-captions roll a window forward and repeat text; the parser must strip
that overlap at word granularity without losing content."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fetch_episodes import coalesce, parse_vtt  # noqa: E402

ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:04.000
so the thing people always ask about

00:00:04.000 --> 00:00:07.000
so the thing people always ask about
is leverage and how liquidation

00:00:07.000 --> 00:00:10.500
is leverage and how liquidation
actually works when the market moves

00:00:10.500 --> 00:00:13.000
actually works when the market moves
against your position
"""


def test_strips_rolling_overlap():
    segs = parse_vtt(ROLLING_VTT)
    joined = " ".join(s["text"] for s in segs)
    # Each distinct phrase appears exactly once — no doubling.
    assert joined.count("is leverage and how liquidation") == 1
    assert joined.count("actually works when the market moves") == 1
    assert "against your position" in joined


def test_keeps_timestamps_ascending():
    segs = parse_vtt(ROLLING_VTT)
    times = [s["t"] for s in segs]
    assert times == sorted(times)
    assert times[0] == 1.0


def test_no_content_lost():
    segs = parse_vtt(ROLLING_VTT)
    joined = " ".join(s["text"] for s in segs)
    for word in ("leverage", "liquidation", "market", "position"):
        assert word in joined


def test_ignores_headers_and_cue_numbers():
    segs = parse_vtt(ROLLING_VTT)
    assert not any("WEBVTT" in s["text"] or "Kind:" in s["text"] for s in segs)


def test_coalesce_merges_tiny_fragments():
    fragments = [
        {"t": 0.0, "text": "one"},
        {"t": 1.0, "text": "two"},     # within min_gap -> merges up
        {"t": 20.0, "text": "three"},  # gap -> new segment
    ]
    out = coalesce(fragments, min_gap=6.0)
    assert len(out) == 2
    assert out[0]["text"] == "one two"
    assert out[0]["t"] == 0.0
    assert out[1]["text"] == "three"


def test_empty_input():
    assert parse_vtt("WEBVTT\n\n") == []
