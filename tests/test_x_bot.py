"""The X mention bot: what it says, what it refuses, and what it spends.

Two things here are worth more than the rest. The link guard, because a URL
in a reply costs 13x and X does not document what counts as one. And the
cold-start behaviour, because a bot that wakes up with no state and answers
a month of old mentions at once is the pattern that gets accounts
suspended — the failure is reputational and not reversible.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import urllib.parse
from dataclasses import dataclass

import pytest

from app.x_api import (
    PRICE_OWNED_READ,
    LinkInReplyError,
    Mention,
    XCredentials,
    assert_linkless,
    strip_urls,
)
from app.x_bot import BotState, MentionBot, format_reply, question_from

# --- reading the question --------------------------------------------------

@pytest.mark.parametrize("post,expected", [
    ("@MarketBubbleAI what did ansem say about eth",
     "what did ansem say about eth"),
    # Every handle goes, not just the bot's: the others are people being
    # looped in, not search terms.
    ("@marketbubble @MarketBubbleAI what did squire say",
     "what did squire say"),
    ("@MarketBubbleAI    spaced   out   question",
     "spaced out question"),
    ("@MarketBubbleAI", ""),
])
def test_question_from(post, expected):
    assert question_from(post) == expected


# --- the link guard --------------------------------------------------------

@pytest.mark.parametrize("text", [
    "see https://search.lexthedev.com",
    "go to www.example.com",
    "it's at search.lexthedev.com",          # bare domain, no scheme
    "the token is on pump.fun",
    "check base.org for details",
])
def test_assert_linkless_rejects_url_shapes(text):
    """Broader than "starts with http" on purpose.

    X publishes a $0.200 charge for a post "with a URL" and does not say
    what their detector counts. Refusing anything URL-shaped keeps the
    $0.015 rate whichever way the rule actually works.
    """
    with pytest.raises(LinkInReplyError):
        assert_linkless(text)


@pytest.mark.parametrize("text", [
    "he said that at 3:52:34 on the luca netz episode",
    "ansem thinks eth is done. 1:04:12 · Market Bubble Ep 12",
    "the price was $77.96 and it moved 0.04% that day",   # decimals are not domains
    "chris gilbert from squire ai protocol came on late",  # 'ai' not preceded by a dot
])
def test_assert_linkless_allows_ordinary_replies(text):
    assert_linkless(text)          # must not raise


def test_strip_urls_cleans_a_quoted_transcript_line():
    """Guests read links aloud and Whisper writes them down."""
    quoted = "go sign up at usepod.io it's a marketplace for inference"
    assert "usepod.io" not in strip_urls(quoted)
    assert "marketplace for inference" in strip_urls(quoted)


# --- the reply itself ------------------------------------------------------

@dataclass
class FakeHit:
    title: str = "LIVE W/ LUCA NETZ & GPT-LIVE: Market Bubble Ep 10"
    timestamp: str = "3:52:34"
    deep_link: str = "https://x.com/MarketBubble/status/2075316750439338088"


def test_reply_cites_the_moment_and_carries_no_link():
    reply = format_reply("Chris Gilbert came on to talk about Squire.",
                         [FakeHit()])
    assert "3:52:34" in reply
    assert "Market Bubble Ep 10" in reply
    assert_linkless(reply)          # raises if a URL slipped in


def test_reply_fits_in_a_post():
    long_answer = ("Ansem explained his reasoning at considerable length. "
                   * 20)
    reply = format_reply(long_answer, [FakeHit()])
    assert len(reply) <= 280
    assert "3:52:34" in reply, "the citation must survive the trim"


def test_reply_with_links_enabled_carries_the_deep_link():
    """The funded variant. Off by default because it costs 13x."""
    reply = format_reply("Chris Gilbert talked about Squire.", [FakeHit()],
                         include_links=True)
    assert FakeHit().deep_link in reply
    with pytest.raises(LinkInReplyError):
        assert_linkless(reply)      # by design: this is the expensive mode


def test_a_url_in_the_answer_never_reaches_the_reply():
    reply = format_reply("He said to go to usepod.io for the beta.",
                         [FakeHit()])
    assert_linkless(reply)


# --- OAuth 1.0a signing ----------------------------------------------------

def test_signature_matches_a_hand_built_base_string():
    """Check the signature base string, not just that bytes come out.

    Signing fails silently — a wrong base string returns a valid-looking
    header and a 401 that reads like bad credentials. So this rebuilds the
    base string by hand from the spec and asserts the HMAC agrees.
    """
    creds = XCredentials("ck", "cs", "at", "as")
    url = "https://api.x.com/2/tweets"
    header = creds.header("POST", url)

    parts = {}
    for item in header[len("OAuth "):].split(", "):
        key, _, value = item.partition("=")
        parts[key] = urllib.parse.unquote(value.strip('"'))

    signed = {k: v for k, v in parts.items() if k != "oauth_signature"}
    joined = "&".join(f"{urllib.parse.quote(k, safe='')}="
                      f"{urllib.parse.quote(v, safe='')}"
                      for k, v in sorted(signed.items()))
    base = "&".join(["POST", urllib.parse.quote(url, safe=""),
                     urllib.parse.quote(joined, safe="")])
    expected = base64.b64encode(
        hmac.new(b"cs&as", base.encode(), hashlib.sha1).digest()).decode()

    assert parts["oauth_signature"] == expected
    assert parts["oauth_signature_method"] == "HMAC-SHA1"
    assert parts["oauth_consumer_key"] == "ck"


def test_query_parameters_are_part_of_the_signature():
    """A GET whose params are left out of the base string returns 401."""
    creds = XCredentials("ck", "cs", "at", "as")
    url = "https://api.x.com/2/users/1/mentions"
    with_params = creds.header("GET", url, {"max_results": "20"})
    without = creds.header("GET", url)

    def signature(header):
        for item in header[len("OAuth "):].split(", "):
            if item.startswith("oauth_signature="):
                return item
        return None

    assert signature(with_params) != signature(without)


# --- state -----------------------------------------------------------------

def test_state_round_trips(tmp_path):
    path = tmp_path / "state.json"
    BotState(since_id="123", replied=["a", "b"], day="2026-08-26",
             replies_today=4, spent_usd=0.12).save(path)
    back = BotState.load(path)
    assert (back.since_id, back.replies_today) == ("123", 4)
    assert back.replied == ["a", "b"]


def test_unreadable_state_does_not_crash_the_bot(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ truncated")
    assert BotState.load(path).since_id is None


def test_the_replied_list_stays_bounded(tmp_path):
    path = tmp_path / "state.json"
    BotState(since_id="1", replied=[str(i) for i in range(2000)]).save(path)
    assert len(BotState.load(path).replied) == 500


def test_day_roll_resets_the_cap(tmp_path):
    state = BotState(day="2026-08-25", replies_today=100)
    state.roll("2026-08-26")
    assert state.replies_today == 0


# --- the loop --------------------------------------------------------------

class FakeClient:
    bot_user_id = "bot"

    def __init__(self, batches):
        self._batches = list(batches)
        self.posted: list[tuple[str, str]] = []
        self.spent_usd = 0.0

    async def mentions(self, since_id=None, limit=20):
        got = self._batches.pop(0) if self._batches else []
        self.spent_usd += len(got) * PRICE_OWNED_READ
        return got

    async def reply(self, text, to_post_id):
        self.posted.append((to_post_id, text))
        return f"reply-to-{to_post_id}"


class FakeIndex:
    def __init__(self, answer="He said it here.", hits=None, refused=False):
        self._answer, self._hits, self._refused = answer, hits, refused
        self.asked: list[str] = []

    async def search(self, query, top_k=None):
        self.asked.append(query)

        class R:
            answer = self._answer
            hits = self._hits if self._hits is not None else [FakeHit()]
            refused = self._refused
        return R()


def mention(mid, text="@bot what did ansem say", author="someone"):
    return Mention(id=mid, text=text, author_id=author, conversation_id=mid)


@pytest.mark.anyio
async def test_cold_start_skips_the_backlog(tmp_path):
    """The suspension-shaped failure.

    Render's disk is ephemeral, so a redeploy can hand the bot an empty
    state file. If that meant "answer everything", a restart would fire a
    month of replies in one burst.
    """
    client = FakeClient([[mention("1"), mention("2"), mention("3")]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    assert await bot.tick("2026-08-26") == 0
    assert client.posted == []
    assert bot.state.since_id == "3"      # but it remembers where it was


@pytest.mark.anyio
async def test_answers_new_mentions_after_the_first_poll(tmp_path):
    client = FakeClient([[mention("1")], [mention("2"), mention("3")]])
    index = FakeIndex()
    bot = MentionBot(client, index, state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")                      # cold start
    assert await bot.tick("2026-08-26") == 2
    assert [pid for pid, _ in client.posted] == ["2", "3"]
    assert index.asked == ["what did ansem say", "what did ansem say"]


@pytest.mark.anyio
async def test_never_answers_the_same_mention_twice(tmp_path):
    repeat = mention("7")
    client = FakeClient([[mention("1")], [repeat], [repeat]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    await bot.tick("2026-08-26")
    await bot.tick("2026-08-26")
    assert len(client.posted) == 1


@pytest.mark.anyio
async def test_never_answers_itself(tmp_path):
    client = FakeClient([[mention("1")], [mention("2", author="bot")]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 0


@pytest.mark.anyio
async def test_daily_cap_stops_a_runaway(tmp_path):
    """The cap is a spend guard, not a politeness setting."""
    client = FakeClient([[mention("0")],
                         [mention(str(i)) for i in range(1, 30)]])
    bot = MentionBot(client, FakeIndex(), daily_reply_cap=5,
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 5
    assert len(client.posted) == 5


@pytest.mark.anyio
async def test_a_bare_tag_with_no_question_is_ignored(tmp_path):
    client = FakeClient([[mention("1")], [mention("2", text="@bot")]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 0
    assert client.posted == []


@pytest.mark.anyio
async def test_a_refusal_means_silence(tmp_path):
    """Saying nothing beats guessing in public."""
    client = FakeClient([[mention("1")], [mention("2")]])
    bot = MentionBot(client, FakeIndex(refused=True),
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 0


@pytest.mark.anyio
async def test_state_survives_a_restart(tmp_path):
    path = tmp_path / "s.json"
    client = FakeClient([[mention("1")], [mention("2")]])
    first = MentionBot(client, FakeIndex(), state_path=path)
    await first.tick("2026-08-26")
    await first.tick("2026-08-26")

    # A redeploy: new object, same disk.
    resumed = MentionBot(FakeClient([[mention("2")]]), FakeIndex(),
                         state_path=path)
    assert await resumed.tick("2026-08-26") == 0, "already answered"


def test_poll_pauses_are_jittered():
    """Perfectly regular intervals are a documented suspension trigger."""
    pauses = {round(MentionBot.pause_seconds(60), 4) for _ in range(50)}
    assert len(pauses) > 40
    assert all(40 <= p <= 90 for p in pauses)


def test_a_402_is_named_rather_than_thrown_raw():
    """Running out of credits is expected, not exceptional.

    Every billed X call answers 402 once the balance is gone. Left as a raw
    HTTPStatusError it surfaces as a traceback that reads like a crash, on a
    bot that is meant to run unattended. It also needs saying that
    GET /2/users/me is NOT billed, so credentials can verify perfectly
    against a zero balance and the first real call still fails — which is
    exactly how this was found.
    """
    import httpx

    from app.x_api import OutOfCreditsError, _raise_if_out_of_credits

    request = httpx.Request("GET", "https://api.x.com/2/users/1/mentions")
    with pytest.raises(OutOfCreditsError) as exc:
        _raise_if_out_of_credits(
            httpx.Response(402, request=request, text="Payment Required"))
    assert "console.x.com" in str(exc.value), "say where to fix it"
    assert "users/me" in str(exc.value), "explain why whoami passed"

    # Anything else must fall through to the normal error handling.
    for code in (200, 401, 403, 429, 500):
        _raise_if_out_of_credits(httpx.Response(code, request=request))
