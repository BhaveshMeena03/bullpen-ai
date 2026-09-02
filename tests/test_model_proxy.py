"""The way back from a proxy, without a deploy.

The podcast search can be pointed at an inference proxy — usepod routes
the same claude-haiku-4-5 at $0.40/$2.00 against Anthropic's $1.00/$5.00,
and settles in USDC, which is the whole reason to bother: topping up a
balance with a card is the thing that actually stops work here.

Everything about that arrangement is somebody else's uptime. A balance
runs out, a token is revoked, a marketplace goes quiet — and every one of
those looks, to a person typing a question, like the site being broken.
So the fallback is not a setting to flip when that happens; it is already
holding the real key and takes over on the failing request.

Two things this pins:

  the real key never goes to the proxy. usepod authenticates by a token
  in the URL path and ignores the header, so sending it would hand a
  third party a working Anthropic key for no benefit at all.

  a lookalike host is not Anthropic. "api.anthropic.com" is a substring
  of api.anthropic.com.evil.example, and a substring test would send the
  key straight there.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.config import anthropic_client_kwargs, get_settings, redact  # noqa: E402


def _settings(base: str):
    return get_settings().model_copy(update={"anthropic_base_url": base})


class TestTheKeyOnlyGoesToAnthropic:
    def test_no_base_url_means_direct_with_the_real_key(self):
        s = _settings("")
        kw = anthropic_client_kwargs(s)
        assert kw == {"api_key": s.anthropic_api_key}
        assert "base_url" not in kw

    def test_a_proxy_gets_a_placeholder(self):
        s = _settings("https://api.usepod.ai/proxy/sometoken")
        kw = anthropic_client_kwargs(s)
        assert kw["base_url"].endswith("/proxy/sometoken")
        assert kw["api_key"] != s.anthropic_api_key

    def test_anthropic_by_its_own_url_still_gets_the_key(self):
        s = _settings("https://api.anthropic.com")
        assert anthropic_client_kwargs(s)["api_key"] == s.anthropic_api_key

    @pytest.mark.parametrize("lookalike", [
        "https://api.anthropic.com.evil.example",
        "https://api.anthropic.com.attacker.test/v1",
        "https://notapi.anthropic.com",
        "https://api.anthropic.com@evil.example",
    ])
    def test_a_lookalike_host_gets_the_placeholder(self, lookalike):
        """Decided on the parsed hostname, not on the string containing it."""
        s = _settings(lookalike)
        assert anthropic_client_kwargs(s)["api_key"] != s.anthropic_api_key


class TestTheTokenIsNeverPrinted:
    def test_it_is_redacted_out_of_anything_logged(self):
        line = ("error posting to https://api.usepod.ai/proxy/abc123secret/"
                "v1/messages: 402")
        assert "abc123secret" not in redact(line)
        assert "/proxy/<token>" in redact(line)

    def test_redaction_happens_before_truncation(self):
        """Cutting a URL to eighty characters can leave the token whole and
        drop only the part that made it look like a URL."""
        line = "https://api.usepod.ai/proxy/" + "s" * 60 + "/v1/messages"
        assert "s" * 60 not in redact(line)[:80]


class TestTheFallbackIsHeldReady:
    def test_a_proxy_means_a_direct_client_is_built_too(self):
        source = (ROOT / "app" / "podcast.py").read_text()
        assert "self._fallback = None" in source
        assert 'if "base_url" in anthropic_client_kwargs(settings)' in source

    def test_both_answer_paths_fall_back(self):
        """The plain endpoint and the SSE one. A proxy failure that only the
        non-streaming path survived would take down the website, which is
        the surface people are actually sent to."""
        source = (ROOT / "app" / "podcast.py").read_text()
        assert source.count("self._proxy_broke(exc)") >= 2

    def test_a_failure_stops_the_proxy_being_retried_every_time(self):
        """An outage should cost one slow answer, not one per visitor."""
        source = (ROOT / "app" / "podcast.py").read_text()
        assert "PROXY_COOLDOWN_SECONDS" in source

    def test_the_stream_does_not_fall_back_once_text_is_flowing(self):
        """Restarting mid-answer would repeat what is already on screen or
        splice two answers together. The fallback is taken while opening."""
        source = (ROOT / "app" / "podcast.py").read_text()
        opened = source.index("entered = await opener.__aenter__()")
        loop = source.index("async for text in stream.text_stream")
        assert opened < loop
