"""Concierge Discord bot — client (mocked HTTP) and output formatting.

Same approach as test_discord_bot.py: the gateway glue in concierge_bot.py
needs a live connection, so the logic that can actually be wrong lives in
concierge_client.py / concierge_format.py and is verified here.
"""

import asyncio

import httpx
import pytest

from discord_bot.concierge_client import ChatClient, ChatError, ChatResult
from discord_bot.concierge_format import (
    build_chat_payload,
    flatten_tables,
    format_sources,
)
from discord_bot.limits import Cooldown, GlobalThrottle
from discord_bot.memory import ConversationMemory


def _client(handler) -> ChatClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return ChatClient("https://backend.test", client=http, timeout=5)


class TestChatClient:
    def test_posts_to_chat_and_parses_answer(self):
        def handler(req):
            assert req.url.path == "/v1/chat"
            return httpx.Response(200, json={
                "answer": "Transfer USDC into your perps account.",
                "sources": [
                    {"id": "1", "metadata": {"title": "Funding Your Wallet"}},
                ],
                "refused": False,
            })
        res = asyncio.run(_client(handler).ask("balance is zero"))
        assert res.answer.startswith("Transfer USDC")
        assert res.sources == ["Funding Your Wallet"]
        assert res.refused is False

    def test_duplicate_source_pages_collapse(self):
        """The retriever returns several chunks per page; a reader wants the
        page named once, not five times."""
        def handler(req):
            return httpx.Response(200, json={
                "answer": "a",
                "sources": [
                    {"id": "1", "metadata": {"title": "Stop Limit"}},
                    {"id": "2", "metadata": {"title": "Stop Limit"}},
                    {"id": "3", "metadata": {"title": "Stop Market"}},
                ],
            })
        res = asyncio.run(_client(handler).ask("q"))
        assert res.sources == ["Stop Limit", "Stop Market"]

    def test_source_without_title_falls_back(self):
        def handler(req):
            return httpx.Response(200, json={
                "answer": "a",
                "sources": [{"id": "abc", "metadata": {"source_id": "bp-faqs"}}],
            })
        assert asyncio.run(_client(handler).ask("q")).sources == ["bp-faqs"]

    def test_refusal_flag_survives(self):
        def handler(req):
            return httpx.Response(200, json={
                "answer": "I don't have that.", "sources": [], "refused": True})
        assert asyncio.run(_client(handler).ask("q")).refused is True

    def test_429_is_a_capacity_error_not_a_retry(self):
        calls = []

        def handler(req):
            calls.append(1)
            return httpx.Response(429, json={"detail": "slow down"})
        with pytest.raises(ChatError) as exc:
            asyncio.run(_client(handler).ask("q"))
        assert exc.value.kind == "http"
        assert len(calls) == 1, "429 must not be retried — it would amplify load"

    def test_missing_answer_field_is_bad_response(self):
        def handler(req):
            return httpx.Response(200, json={"sources": []})
        with pytest.raises(ChatError) as exc:
            asyncio.run(_client(handler).ask("q"))
        assert exc.value.kind == "bad_response"

    def test_timeout_retries_once_then_raises(self):
        calls = []

        def handler(req):
            calls.append(1)
            raise httpx.TimeoutException("too slow")
        with pytest.raises(ChatError) as exc:
            asyncio.run(_client(handler).ask("q"))
        assert exc.value.kind == "timeout"
        assert len(calls) == 2, "should attempt once, retry once"

    def test_transient_failure_then_success(self):
        calls = []

        def handler(req):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.TimeoutException("cold start")
            return httpx.Response(200, json={"answer": "ok", "sources": []})
        assert asyncio.run(_client(handler).ask("q")).answer == "ok"


class TestFormatting:
    def test_urls_in_answers_are_stripped(self):
        """The concierge answers from docs and must never post a clickable
        link, so a URL reaching the answer body is stripped regardless."""
        res = ChatResult(answer="Go to https://evil.example/drain now")
        out = build_chat_payload("q", res)
        assert "evil.example" not in out["description"]
        assert "[link removed]" in out["description"]

    def test_url_in_the_question_is_defanged_in_the_title(self):
        res = ChatResult(answer="fine")
        out = build_chat_payload("check phishing.xyz/claim", res)
        assert "phishing.xyz" not in out["title"]

    def test_refusal_renders_the_dont_know_message(self):
        out = build_chat_payload("q", ChatResult(answer="", refused=True))
        assert out["empty"] is True
        assert "don't have that" in out["description"]

    def test_empty_answer_treated_as_refusal(self):
        out = build_chat_payload("q", ChatResult(answer="   "))
        assert out["empty"] is True

    def test_sources_are_capped(self):
        titles = [f"Page {i}" for i in range(20)]
        assert format_sources(titles).count("`") == 10  # 5 titles, 2 ticks each

    def test_source_title_that_looks_like_a_domain_is_defanged(self):
        assert "bullpen.fi" not in format_sources(["bullpen.fi"])

    def test_very_long_answer_is_still_truncated(self):
        out = build_chat_payload("q", ChatResult(answer="word " * 2000))
        assert out["description"].endswith("…")


class TestTablesAndLength:
    def test_markdown_table_becomes_plain_lines(self):
        """Discord renders no tables, so a table arrives as a wall of pipes."""
        answer = (
            "Here's what you need:\n"
            "| Market | What you need |\n"
            "|---|---|\n"
            "| Solana Spot | SOL for gas |\n"
            "| Perps | USDC |\n"
            "Done."
        )
        out = flatten_tables(answer)
        assert "|" not in out
        assert "• **Solana Spot** — SOL for gas" in out
        assert "• **Perps** — USDC" in out
        assert out.startswith("Here's what you need:")
        assert out.rstrip().endswith("Done.")

    def test_prose_with_pipes_is_left_alone(self):
        assert flatten_tables("use a | b for piping") == "use a | b for piping"

    def test_support_length_answer_is_not_truncated(self):
        """1400 came from podcast search; support answers run ~2000 and were
        losing their last third."""
        answer = "word " * 400          # ~2000 chars
        out = build_chat_payload("q", ChatResult(answer=answer))
        assert not out["description"].endswith("…")

    def test_still_capped_below_discord_embed_limit(self):
        out = build_chat_payload("q", ChatResult(answer="word " * 2000))
        assert len(out["description"]) <= 3501
        assert len(out["description"]) < 4096, "Discord rejects over 4096"


class TestClientHistory:
    def test_history_is_sent_when_present(self):
        seen = {}

        def handler(req):
            import json
            seen.update(json.loads(req.content))
            return httpx.Response(200, json={"answer": "ok", "sources": []})
        past = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]
        asyncio.run(_client(handler).ask("follow up", history=past))
        assert seen["history"] == past

    def test_history_key_omitted_when_empty(self):
        seen = {}

        def handler(req):
            import json
            seen.update(json.loads(req.content))
            return httpx.Response(200, json={"answer": "ok", "sources": []})
        asyncio.run(_client(handler).ask("first question"))
        assert "history" not in seen


class TestConversationMemory:
    def test_follow_up_sees_the_previous_exchange(self):
        m = ConversationMemory()
        m.append(1, 10, "balance is zero", "which product?", now=100.0)
        turns = m.get(1, 10, now=105.0)
        assert [t["role"] for t in turns] == ["user", "assistant"]
        assert turns[0]["content"] == "balance is zero"

    def test_expires_after_silence(self):
        m = ConversationMemory(ttl=60)
        m.append(1, 10, "q", "a", now=100.0)
        assert m.get(1, 10, now=150.0)          # still inside the window
        assert m.get(1, 10, now=200.0) == []    # gone

    def test_channels_do_not_bleed_into_each_other(self):
        m = ConversationMemory()
        m.append(1, 10, "q", "a", now=100.0)
        assert m.get(1, 11, now=100.0) == []

    def test_users_do_not_see_each_other(self):
        m = ConversationMemory()
        m.append(1, 10, "my balance", "…", now=100.0)
        assert m.get(2, 10, now=100.0) == []

    def test_only_recent_turns_kept(self):
        """History is resent every request, so unbounded history is an
        unbounded bill."""
        m = ConversationMemory(max_turns=4)
        for i in range(5):
            m.append(1, 10, f"q{i}", f"a{i}", now=100.0 + i)
        turns = m.get(1, 10, now=110.0)
        assert len(turns) == 4
        assert turns[-1]["content"] == "a4"
        assert all("q0" != t["content"] for t in turns)

    def test_conversation_count_is_bounded(self):
        m = ConversationMemory(max_conversations=3)
        for uid in range(10):
            m.append(uid, 10, "q", "a", now=100.0 + uid)
        assert len(m._store) <= 3

    def test_expired_entries_are_reclaimed_not_just_hidden(self):
        m = ConversationMemory(ttl=10, max_conversations=100)
        m.append(1, 10, "q", "a", now=100.0)
        m.append(2, 10, "q", "a", now=500.0)   # triggers eviction sweep
        assert (1, 10) not in m._store

    def test_forget_clears_one_conversation(self):
        m = ConversationMemory()
        m.append(1, 10, "q", "a", now=100.0)
        assert m.forget(1, 10) is True
        assert m.get(1, 10, now=101.0) == []
        assert m.forget(1, 10) is False


class TestSharedLimits:
    """These moved out of bot.py so both bots share one implementation."""

    def test_cooldown_blocks_then_expires(self):
        cd = Cooldown(10)
        cd.stamp(1, now=100.0)
        assert cd.retry_after(1, now=105.0) == pytest.approx(5.0)
        assert cd.retry_after(1, now=111.0) == 0.0

    def test_cooldown_is_per_user(self):
        cd = Cooldown(10)
        cd.stamp(1, now=100.0)
        assert cd.retry_after(2, now=100.0) == 0.0

    def test_global_throttle_caps_a_burst(self):
        t = GlobalThrottle(per_min=5)
        allowed = sum(1 for _ in range(20) if t.allow(now=1000.0))
        assert allowed == 5, "a raid must not blow past the bot-wide ceiling"

    def test_global_throttle_refills_over_time(self):
        t = GlobalThrottle(per_min=60)  # one per second
        for _ in range(60):
            t.allow(now=1000.0)
        assert t.allow(now=1000.5) is False
        assert t.allow(now=1002.0) is True


class TestBriefRequest:
    def test_discord_always_asks_for_a_brief_answer(self):
        seen = {}

        def handler(req):
            import json
            seen.update(json.loads(req.content))
            return httpx.Response(200, json={"answer": "ok", "sources": []})
        asyncio.run(_client(handler).ask("how do i set a stop loss?"))
        assert seen["brief"] is True


class TestDetailedOverride:
    """Brief is the right default for a chat window, but a user who wants the
    long answer must be able to get it — the knowledge is the same, only the
    length differs, so the choice belongs to whoever is reading."""

    @staticmethod
    def _payload_for(brief: bool) -> dict:
        seen = {}

        def handler(req):
            import json
            seen.update(json.loads(req.content))
            return httpx.Response(200, json={"answer": "ok", "sources": []})
        asyncio.run(_client(handler).ask("q", brief=brief))
        return seen

    def test_default_is_brief(self):
        assert self._payload_for(True)["brief"] is True

    def test_detailed_turns_brief_off(self):
        assert self._payload_for(False)["brief"] is False
