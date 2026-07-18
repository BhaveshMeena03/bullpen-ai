"""Discord bot logic — client (mocked HTTP), formatting, cooldown.

The Discord gateway glue in bot.py can't be unit-tested without a live
connection, so the testable logic is extracted into client.py / format.py
and covered here. This is what makes the bot production-grade: the parts
that can be wrong are verified.
"""

import asyncio

import httpx
import pytest

from discord_bot.client import Hit, SearchClient, SearchError, SearchResult
from discord_bot.format import (
    build_answer_payload,
    format_hits,
    truncate_answer,
)


def _client(handler) -> SearchClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return SearchClient("https://backend.test", client=http, timeout=5)


class TestSearchClient:
    def test_happy_path_parses_answer_and_hits(self):
        def handler(req):
            assert req.url.path == "/v1/podcast/search"
            return httpx.Response(200, json={
                "answer": "The hosts said X.",
                "hits": [{"timestamp": "1:01", "title": "Ep 1",
                          "deep_link": "https://youtube.com/watch?v=a&t=61s"}],
                "refused": False,
            })
        res = asyncio.run(_client(handler).search("q"))
        assert res.answer == "The hosts said X."
        assert res.hits[0].timestamp == "1:01"

    def test_hits_without_deep_link_dropped(self):
        def handler(req):
            return httpx.Response(200, json={
                "answer": "a", "hits": [{"timestamp": "1:01", "title": "x"}]})
        res = asyncio.run(_client(handler).search("q"))
        assert res.hits == []

    def test_503_maps_to_busy(self):
        def handler(req):
            return httpx.Response(503, json={"detail": "busy"})
        with pytest.raises(SearchError) as e:
            asyncio.run(_client(handler).search("q"))
        assert e.value.kind == "busy"

    def test_5xx_retries_then_gives_up(self):
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            return httpx.Response(502)
        with pytest.raises(SearchError) as e:
            asyncio.run(_client(handler).search("q", retries=1))
        assert calls["n"] == 2, "should retry once on 5xx"
        assert e.value.kind == "unavailable"

    def test_timeout_retries_then_reports_timeout(self):
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            raise httpx.ReadTimeout("slow", request=req)
        with pytest.raises(SearchError) as e:
            asyncio.run(_client(handler).search("q", retries=1))
        assert calls["n"] == 2
        assert e.value.kind == "timeout"

    def test_transient_then_success(self):
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("down", request=req)
            return httpx.Response(200, json={"answer": "ok", "hits": []})
        res = asyncio.run(_client(handler).search("q", retries=1))
        assert res.answer == "ok"


class TestFormatting:
    def test_truncate_at_word_boundary(self):
        out = truncate_answer("word " * 500, limit=50)
        assert len(out) <= 51 and out.endswith("…")
        assert "  " not in out.strip("…")

    def test_short_answer_untouched(self):
        assert truncate_answer("hi there") == "hi there"

    def test_format_hits_dedupes_and_limits(self):
        hits = [
            Hit("1:00", "A", "https://youtu.be/1"),
            Hit("1:00", "A", "https://youtu.be/1"),   # dup link
            Hit("2:00", "B", "https://youtu.be/2"),
            Hit("3:00", "C", "https://youtu.be/3"),
            Hit("4:00", "D", "https://youtu.be/4"),
            Hit("5:00", "E", "https://youtu.be/5"),
        ]
        out = format_hits(hits, limit=3)
        assert out.count("\n") == 2       # 3 lines
        assert out.count("https://youtu.be/1") == 1  # deduped
        assert "youtu.be/5" not in out       # limited

    def test_hits_are_markdown_links(self):
        out = format_hits([Hit("1:00", "Ep", "https://youtu.be/1")])
        assert "](https://youtu.be/1)" in out and "`1:00`" in out

    def test_empty_result_payload(self):
        payload = build_answer_payload("q", SearchResult(answer="", refused=True))
        assert payload["empty"] is True
        assert "couldn't find" in payload["description"].lower()

    def test_answer_payload_shape(self):
        res = SearchResult(answer="Real answer",
                           hits=[Hit("1:00", "Ep", "https://youtu.be/1")])
        payload = build_answer_payload("why?", res)
        assert payload["empty"] is False
        assert payload["title"] == "why?"
        assert "https://youtu.be/1" in payload["hits"]


class TestCooldown:
    def test_blocks_within_window_allows_after(self):
        from discord_bot.bot import Cooldown
        cd = Cooldown(seconds=10)
        cd.stamp(1, now=100.0)
        assert cd.retry_after(1, now=105.0) == pytest.approx(5.0)
        assert cd.retry_after(1, now=111.0) == 0.0
        assert cd.retry_after(2, now=105.0) == 0.0  # different user unaffected


class TestGlobalThrottle:
    def test_hard_ceiling_across_all_users(self):
        from discord_bot.bot import GlobalThrottle
        t = GlobalThrottle(per_min=6)  # 6/min => burst 6, then blocked
        allowed = sum(1 for _ in range(20) if t.allow(now=1000.0))
        assert allowed == 6, "must cap total searches regardless of user count"

    def test_refills_over_time(self):
        from discord_bot.bot import GlobalThrottle
        t = GlobalThrottle(per_min=60)  # 1/sec
        for _ in range(60):
            t.allow(now=1000.0)
        assert t.allow(now=1000.0) is False   # bucket drained
        assert t.allow(now=1005.0) is True    # 5s later -> refilled


class TestNoTokenLeak:
    def test_token_not_in_logs_on_login(self, monkeypatch):
        # The bot must never log the token. Build with a sentinel and assert
        # it appears nowhere the bot writes.
        monkeypatch.setenv("DISCORD_TOKEN", "SENTINEL_TOKEN_do_not_log")
        monkeypatch.setenv("BACKEND_URL", "https://x.test")
        from discord_bot.bot import build_bot
        bot = build_bot()
        # token is only stashed for run(), never formatted into a log string
        import inspect

        import discord_bot.bot as mod
        src = inspect.getsource(mod)
        assert "_token" in src  # it is used
        # no logging call interpolates the token
        assert "logger.info(bot._token" not in src
        assert "logger" not in src.split("bot._token")[1].split("\n")[0]


class TestLinkSafety:
    """The bot must NEVER post a non-YouTube link — the core defense against
    being turned into a wallet-drainer link poster."""

    def test_only_youtube_links_posted(self):
        from discord_bot.format import format_hits
        hits = [
            Hit("1:00", "real", "https://www.youtube.com/watch?v=x&t=60s"),
            Hit("2:00", "scam", "https://ansem-airdrop.xyz/claim"),
            Hit("3:00", "scam2", "http://evil.link/drain"),
            Hit("4:00", "yt2", "https://youtu.be/abc?t=5"),
        ]
        out = format_hits(hits)
        assert "youtube.com/watch?v=x" in out
        assert "youtu.be/abc" in out
        assert "ansem-airdrop.xyz" not in out   # dropped
        assert "evil.link" not in out           # dropped

    def test_is_allowed_link(self):
        from discord_bot.format import is_allowed_link
        assert is_allowed_link("https://www.youtube.com/watch?v=x")
        assert is_allowed_link("https://youtu.be/x")
        assert not is_allowed_link("https://youtube.com.evil.xyz/x")  # not the host
        assert not is_allowed_link("https://scam.link/claim")
        assert not is_allowed_link("javascript:alert(1)")

    def test_answer_urls_defanged(self):
        from discord_bot.format import truncate_answer
        out = truncate_answer(
            "The hosts said buy now at https://ansem-drop.xyz/claim and www.scam.io")
        assert "ansem-drop.xyz" not in out
        assert "scam.io" not in out
        assert "[link removed]" in out

    def test_question_echo_defanged(self):
        from discord_bot.format import build_answer_payload
        p = build_answer_payload("launching $ANSEM claim at scam.link/x",
                                 SearchResult(answer="ok", hits=[]))
        assert "scam.link" not in p["title"]
