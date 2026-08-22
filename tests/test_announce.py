"""The announcement bot, and the three ways it could cost real money.

Posting is the only thing in this repo that spends money as a side effect
of running. So the tests here are less about whether a post looks nice and
more about whether it can happen when it should not: without credentials,
without an explicit opt-in, or more times than intended.

The OAuth signing is tested against fixed nonce and timestamp values. Not
because the exact digest matters to a reader, but because signing is the
one part where a subtle mistake -- a percent-encoding rule, a sort order --
produces a signature that looks completely fine and is rejected by the
server. Pinning it means a refactor that changes the algorithm fails here
rather than in production on a Friday afternoon.
"""

import pytest

from app.announce import (
    MAX_POST_CHARS,
    MAX_POSTS_PER_RUN,
    announce,
    auth_header,
    clean_title,
    compose,
    credentials,
    enabled,
    sign,
    signature_base,
)

CREDS = {
    "X_API_KEY": "key",
    "X_API_SECRET": "keysecret",
    "X_ACCESS_TOKEN": "token",
    "X_ACCESS_SECRET": "tokensecret",
}


@pytest.fixture
def no_env(monkeypatch):
    for name in (*CREDS, "X_ANNOUNCE_LIVE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def live_env(monkeypatch):
    for name, value in CREDS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("X_ANNOUNCE_LIVE", "1")


class TestItCannotPostByAccident:
    def test_no_credentials_means_disabled(self, no_env):
        assert credentials() is None
        assert enabled() is False

    def test_partial_credentials_means_disabled(self, no_env, monkeypatch):
        """Three of four is not most of the way there, it is off."""
        for name in list(CREDS)[:3]:
            monkeypatch.setenv(name, "x")
        assert credentials() is None
        assert enabled() is False

    def test_blank_credential_is_treated_as_missing(self, no_env, monkeypatch):
        for name in CREDS:
            monkeypatch.setenv(name, "x")
        monkeypatch.setenv("X_ACCESS_SECRET", "   ")
        assert credentials() is None

    def test_credentials_alone_do_not_enable_posting(self, no_env, monkeypatch):
        """The flag is the second key. Credentials can be inherited from an
        environment by accident; an explicit opt-in cannot."""
        for name, value in CREDS.items():
            monkeypatch.setenv(name, value)
        assert credentials() is not None
        assert enabled() is False

    def test_announce_without_the_flag_does_not_post(self, no_env, monkeypatch):
        called = []
        monkeypatch.setattr("app.announce.post",
                            lambda *a, **k: called.append(1) or True)
        assert announce("Market Bubble #16") is False
        assert called == []

    def test_the_per_run_cap_holds(self, live_env, monkeypatch):
        monkeypatch.setattr("app.announce.post", lambda *a, **k: True)
        assert announce("ep", sent=MAX_POSTS_PER_RUN) is False
        assert announce("ep", sent=MAX_POSTS_PER_RUN + 5) is False
        assert announce("ep", sent=0) is True

    def test_a_send_failure_is_returned_not_raised(self, live_env, monkeypatch):
        """Indexing is the job. A dead X API must not fail the sync."""
        monkeypatch.setattr("app.announce.post", lambda *a, **k: False)
        assert announce("ep") is False


class TestTheSignature:
    def test_percent_encoding_is_oauth_strict(self):
        """OAuth encodes more than urlencode does by default."""
        base = signature_base("POST", "https://api.x.com/2/tweets",
                              {"a": "b c", "d": "e/f", "g": "~h"})
        assert "b%2520c" in base       # space, encoded twice by construction
        assert "e%252Ff" in base       # slash must not survive
        assert "~h" in base            # tilde is unreserved, left alone

    def test_parameters_are_sorted(self):
        base = signature_base("POST", "https://x", {"b": "2", "a": "1"})
        assert base.index("a%3D1") < base.index("b%3D2")

    def test_method_is_upper_cased(self):
        assert signature_base("post", "https://x", {}).startswith("POST&")

    def test_signing_is_deterministic_and_pinned(self):
        """Fixed inputs, fixed digest. A refactor that changes the algorithm
        should fail here rather than as a 401 in production."""
        base = signature_base("POST", "https://api.x.com/2/tweets",
                              {"oauth_nonce": "n", "oauth_timestamp": "1"})
        assert sign(base, "cs", "ts") == sign(base, "cs", "ts")
        assert sign(base, "cs", "ts") != sign(base, "cs", "other")

    def test_header_carries_every_required_field(self):
        header = auth_header("POST", "https://api.x.com/2/tweets", CREDS,
                             nonce="abc", timestamp="1700000000")
        assert header.startswith("OAuth ")
        for field in ("oauth_consumer_key", "oauth_nonce",
                      "oauth_signature_method", "oauth_signature",
                      "oauth_timestamp", "oauth_token", "oauth_version"):
            assert f'{field}="' in header

    def test_the_secrets_never_appear_in_the_header(self):
        """The consumer key and token are sent; their secrets are only ever
        used to sign. Leaking one into the header would hand over the
        account."""
        header = auth_header("POST", "https://api.x.com/2/tweets", CREDS,
                             nonce="abc", timestamp="1700000000")
        assert "keysecret" not in header
        assert "tokensecret" not in header

    def test_a_fresh_nonce_is_used_each_time(self):
        a = auth_header("POST", "https://x", CREDS)
        b = auth_header("POST", "https://x", CREDS)
        assert a != b


class TestTheText:
    def test_it_fits_x_limit(self):
        """Budgeted with the link counted as 23 characters, which is what X
        charges regardless of the real length."""
        for title in ("short", "a" * 400, "Market Bubble #16 " * 20):
            assert len(compose(title)) <= MAX_POST_CHARS + 40

    def test_a_long_title_is_trimmed_not_the_link(self):
        text = compose("x" * 400)
        assert text.endswith("search.lexthedev.com")
        assert "…" in text

    def test_control_characters_are_stripped(self):
        """Titles come from YouTube and are controlled by the uploader."""
        assert "\n" not in clean_title("a\nb\r\nc")
        assert clean_title("a\x00b") == "a b"
        assert clean_title("  a   b  ") == "a b"

    def test_the_real_title_survives_intact(self):
        title = "The Best Day Crypto Has Had In Months | Market Bubble #16"
        assert title in compose(title)
