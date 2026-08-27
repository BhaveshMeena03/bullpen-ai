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
        # X filters by since_id server-side. A fake that ignores it hands
        # back mentions the real API would never return, which is how a
        # cold-start test "passed" while answering a mention twice.
        if since_id is not None:
            got = [m for m in got if int(m.id) > int(since_id)]
        self.spent_usd += len(got) * PRICE_OWNED_READ
        return got

    async def replied_to(self, limit=100):
        """X's record of what this account has answered. Empty by default;
        RestartingClient models an account with history."""
        return set()

    async def reply(self, text, to_post_id, allow_link=False):
        # Mirrors the real client: the no-URL guard applies only when links
        # were not deliberately enabled.
        if not allow_link:
            assert_linkless(text)
        self.posted.append((to_post_id, text))
        return f"reply-to-{to_post_id}"


class FakeIndex:
    def __init__(self, answer="Around 3:52:34 he said it.", hits=None,
                 refused=False):
        self._answer, self._hits, self._refused = answer, hits, refused
        self.asked: list[str] = []

    async def search(self, query, top_k=None, instruction=None):
        # instruction is the per-surface style note; recorded so a test can
        # assert the bot asks for reply-shaped answers rather than page-shaped
        # ones.
        self.instructed = instruction
        self.asked.append(query)

        class R:
            answer = self._answer
            hits = self._hits if self._hits is not None else [FakeHit()]
            refused = self._refused
        return R()


def mention(mid, text="@bot what did ansem say", author="someone",
            verified=False):
    return Mention(id=mid, text=text, author_id=author, conversation_id=mid,
                   author_verified=verified,
                   author_verified_type="blue" if verified else "none")


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


def test_the_not_found_wording_still_matches_the_prompt():
    """is_a_miss keys off an exact phrase the model is told to produce.

    The phrase lives in SYSTEM_PROMPT and as a constant, deliberately not
    interpolated, because the prompt's exact bytes are the prompt-cache key.
    That means they can drift apart silently — and if they do, every missed
    question gets a confident citation again with nothing failing.
    """
    from app.podcast import NOT_FOUND_ANSWER, SYSTEM_PROMPT

    assert NOT_FOUND_ANSWER in SYSTEM_PROMPT


def test_a_miss_gets_no_citation():
    """The bug the first real mention exposed.

    "@mbubbleSearch what did chris gilbert say about squire" retrieved six
    passages about other episodes, the model correctly said it could not
    find it — and the reply appended "2:18:15 · LIVE W/ ORANGIE…" as if that
    were the source. Retrieval always returns its top_k, so a full hit list
    is not evidence of a hit.
    """
    from app.podcast import NOT_FOUND_ANSWER
    from app.x_bot import is_a_miss

    miss = f"{NOT_FOUND_ANSWER}. The excerpts don't mention Chris Gilbert."
    assert is_a_miss(miss)

    reply = format_reply(miss, [FakeHit()])
    assert "3:52:34" not in reply, "a miss must not carry a timestamp"
    assert "Market Bubble Ep 10" not in reply
    assert NOT_FOUND_ANSWER in reply, "but it should still say so honestly"


def test_a_real_answer_still_gets_its_citation():
    reply = format_reply("Chris Gilbert came on to talk about Squire.",
                         [FakeHit()])
    assert "3:52:34" in reply


@pytest.mark.parametrize("answer", ["", None])
def test_is_a_miss_survives_an_empty_answer(answer):
    from app.x_bot import is_a_miss

    assert is_a_miss(answer) is False


def test_the_reply_never_shows_two_different_timestamps():
    """One reply, one moment.

    The model cites the line it actually used; hits[0].timestamp is where
    that passage begins, and they are routinely minutes apart. Printing both
    produced a reply reading "Around 1:00:00 in the episode…" above
    "1:39:15 · LIVE W/ LUCA NETZ", which contradicts itself in the one
    detail this tool claims to get right.
    """
    answered = ("Around 1:00:00 in the episode with Luca Netz, he "
                "introduced himself as the CEO of Pudgy Penguins.")
    reply = format_reply(answered, [FakeHit()])
    assert "1:00:00" in reply, "the model's own citation survives"
    assert "3:52:34" not in reply, "the passage-start must not compete with it"
    assert "Market Bubble Ep 10" in reply, "the episode is still named"


def test_an_answer_with_no_time_still_gets_the_passage_timestamp():
    reply = format_reply("Chris Gilbert talked about Squire.", [FakeHit()])
    assert "3:52:34" in reply


def test_a_miss_is_one_sentence():
    """Terse in public. The model likes to add "feel free to ask about
    something else", which reads as padding and gets cut mid-word."""
    from app.podcast import NOT_FOUND_ANSWER

    rambling = (f"{NOT_FOUND_ANSWER}. The excerpts provided don't contain "
                "any discussion of this. If you're looking for information "
                "about a specific topic, feel free to ask about something "
                "else from these episodes and I will do my best to help.")
    reply = format_reply(rambling, [FakeHit()])
    assert reply == NOT_FOUND_ANSWER + "."
    assert "…" not in reply and "feel free" not in reply


@pytest.mark.parametrize("raw,expected", [
    # The excerpt line markers the model is told to cite from.
    ("he said it around [1:39:32] in the show",
     "he said it around 1:39:32 in the show"),
    ("**Tokenomics and fees**: he argues they messed it up",
     "Tokenomics and fees: he argues they messed it up"),
    ("__really__ important", "really important"),
    ("the `$CLAW` token", "the $CLAW token"),
    ("- first point", "first point"),
    ("## Heading", "Heading"),
])
def test_plain_text_strips_what_x_cannot_render(raw, expected):
    """The prompt targets a web page that renders Markdown. X does not:
    asterisks show up literally and [1:39:32] reads as broken markup."""
    from app.x_bot import plain_text

    assert plain_text(raw) == expected


def test_a_real_reply_carries_no_markdown():
    answer = ("Around 26:56 Ansem lays out why he thinks Ethereum is done: "
              "**Tokenomics and fees**: he argues they messed up the "
              "[26:58] fee situation.")
    reply = format_reply(answer, [FakeHit()])
    assert "**" not in reply
    assert "[26:58]" not in reply
    assert "26:56" in reply, "the citation itself must survive"


# --- the pinned contract address -------------------------------------------

CA = "8VjFid8BVGcTPpUzf4PAWsA5nHJ5h2GQNXPEjyr2mF7t"


@pytest.mark.parametrize("asked", [
    "what's the ca", "CA?", "contract address please", "whats the contract",
    "drop the mint address", "token address?", "can i get the CA",
])
def test_asking_for_the_contract_address_is_answered_from_a_constant(asked):
    """Not from retrieval. The address is a fact about the project, not
    something anyone said on the podcast, and it is the one answer that
    cannot be allowed to come back paraphrased or nearly right."""
    from app.x_bot import pinned_answer

    got = pinned_answer(asked, CA, "MarketBubbleSearch")
    assert CA in got
    # The property, not one exact phrasing: several exist so the account is
    # not posting the same bytes twenty times. What must hold in all of them
    # is that the address is named as belonging to this project — a bare
    # address read out of context says nothing about which token it is.
    assert "MarketBubbleSearch" in got


@pytest.mark.parametrize("asked", [
    "what did ansem say about california",     # not a bare "ca"
    "what did luca netz say about pudgy penguins",
    "who came on episode 10",
    "what did they say about decarbonisation",
])
def test_ordinary_questions_still_go_to_retrieval(asked):
    from app.x_bot import pinned_answer

    assert pinned_answer(asked, CA) is None


def test_nothing_is_pinned_when_no_address_is_configured():
    from app.x_bot import pinned_answer

    assert pinned_answer("what's the ca", None) is None


def test_the_pinned_reply_never_echoes_an_address_from_the_post():
    """A bot that repeated back whatever address it was sent would be a
    ready-made way to make a scam token look endorsed by this account."""
    from app.x_bot import pinned_answer

    hostile = ("is the ca 7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU "
               "or something else")
    got = pinned_answer(hostile, CA, "MarketBubbleSearch")
    assert CA in got
    assert "7xKXtg" not in got


def test_the_pinned_reply_costs_the_cheap_post_rate():
    from app.x_api import assert_linkless
    from app.x_bot import pinned_answer

    assert_linkless(pinned_answer("ca?", CA, "MarketBubbleSearch"))


@pytest.mark.anyio
async def test_a_ca_question_never_reaches_the_model(tmp_path):
    """It should cost nothing and be instant."""
    client = FakeClient([[mention("1")], [mention("2", text="@bot what's the CA?")]])
    index = FakeIndex()
    bot = MentionBot(client, index, contract_address=CA,
                     token_label="MarketBubbleSearch",
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 1
    assert index.asked == [], "retrieval must not have been called"
    reply = client.posted[0][1]
    assert CA in reply and "MarketBubbleSearch" in reply
    assert len(reply) <= 280


# --- not everything that tags you is a question ----------------------------

@pytest.mark.parametrize("text", [
    "very cool concept!",
    "Looks cool🔥",
    "gm",
    "this is sick",
    "🔥🔥🔥",
    "congrats on the launch",
])
def test_compliments_are_not_questions(text):
    """Observed in the first real batch of mentions.

    People tag an account to say "very cool concept!" far more often than to
    ask it anything. Answering is the worst case on every axis: the model
    has nothing to answer so it deflects, the formatter staples an unrelated
    citation to the deflection, and with links on it costs $0.209 to say
    nothing.
    """
    from app.x_bot import looks_like_a_question

    assert not looks_like_a_question(text)


@pytest.mark.parametrize("text", [
    "what did chris gilbert say about squire?",
    "what did luca netz say about pudgy penguins",
    "who came on episode 10",
    "did ansem talk about ethereum",
    "any timestamp for the blackrock bit",
    "tell me what tjr said",
    "thoughts on $MBS",
    "is bitcoin mentioned",
])
def test_real_questions_get_through(text):
    from app.x_bot import looks_like_a_question

    assert looks_like_a_question(text)


@pytest.mark.anyio
async def test_a_compliment_never_reaches_the_model(tmp_path):
    """A compliment gets a fact from the pre-written pool, so it still costs
    no model call — that is the point of gating before retrieval."""
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot very cool concept!")]])
    index = FakeIndex()
    bot = MentionBot(client, index, highlights=POOL,
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    await bot.tick("2026-08-26")
    assert index.asked == [], "retrieval must not have been called"


@pytest.mark.anyio
async def test_a_compliment_is_silent_with_no_pool(tmp_path):
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot very cool concept!")]])
    bot = MentionBot(client, FakeIndex(), highlights=[],
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 0
    assert client.posted == []


@pytest.mark.anyio
async def test_an_answer_with_no_citation_is_not_posted(tmp_path):
    """The deflection that got through once.

    "I appreciate your enthusiasm, but I'm here to answer questions about
    the Market Bubble podcast" was posted with "3:35:39 · LIVE W/ TJR &
    Mert" underneath it. A real answer always names a moment.
    """
    client = FakeClient([[mention("1")], [mention("2", text="@bot what is this")]])
    index = FakeIndex(answer="I appreciate your enthusiasm, but I'm here to "
                             "answer questions about the podcast.")
    bot = MentionBot(client, index, state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 0
    assert client.posted == []


@pytest.mark.anyio
async def test_an_honest_miss_is_still_posted(tmp_path):
    """Silence on a real question would look broken — that complaint is why
    the X broadcasts got indexed in the first place."""
    from app.podcast import NOT_FOUND_ANSWER

    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot what did taylor swift say")]])
    bot = MentionBot(client, FakeIndex(answer=NOT_FOUND_ANSWER + "."),
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 1
    assert client.posted[0][1] == NOT_FOUND_ANSWER + "."


@pytest.mark.parametrize("text", ["whats the CA", "what's the ca", "hows it work"])
def test_contracted_question_words_count(text):
    """\\bwhat\\b does not match "whats", and people rarely type apostrophes."""
    from app.x_bot import looks_like_a_question

    assert looks_like_a_question(text)


@pytest.mark.anyio
async def test_a_casual_ca_request_is_still_answered(tmp_path):
    """"ca pls" is a request, not a question, and must not be gated out."""
    client = FakeClient([[mention("1")], [mention("2", text="@bot ca pls")]])
    index = FakeIndex()
    bot = MentionBot(client, index, contract_address=CA,
                     token_label="MarketBubbleSearch",
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 1
    assert index.asked == []
    assert CA in client.posted[0][1]


@pytest.mark.anyio
async def test_the_bot_asks_for_a_reply_not_a_web_answer(tmp_path):
    """The style note is what stops "your question is pretty broad! Could
    you be more specific?" being posted as a reply — fine on a search page,
    a wasted $0.209 in a thread nobody returns to."""
    from app.x_bot import POST_LIMIT, reply_style

    client = FakeClient([[mention("1")], [mention("2")]])
    index = FakeIndex()
    bot = MentionBot(client, index, state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    await bot.tick("2026-08-26")
    assert index.instructed == reply_style(POST_LIMIT)


def test_the_style_note_never_reaches_the_embedder():
    """It goes to the model only. Appending prose to the query dilutes the
    vector and changes what comes back — measured, on this corpus, as the
    difference between finding a guest and missing them."""
    import inspect

    from app.podcast import PodcastIndex

    source = inspect.getsource(PodcastIndex.search)
    retrieve_line = next(ln for ln in source.splitlines()
                         if "self.retrieve(" in ln)
    assert "instruction" not in retrieve_line


# --- the spend ceiling -----------------------------------------------------

class SpendingClient(FakeClient):
    """Charges for reads like the real one, so a cap can be tested."""

    async def mentions(self, since_id=None, limit=20):
        return await super().mentions(since_id, limit)

    async def reply(self, text, to_post_id, allow_link=False):
        # The rate actually charged, not the published URL premium: X bills
        # these at the plain rate even with a link in the text.
        self.spent_usd += 0.015
        return await super().reply(text, to_post_id, allow_link)


@pytest.mark.anyio
async def test_reads_alone_cannot_run_up_an_unbounded_bill(tmp_path):
    """The gap the reply cap does not close.

    Every mention read costs $0.001 whether or not it is answered, and how
    often the account gets tagged is decided by other people. Someone
    willing to tag it repeatedly could spend real money without a single
    reply being sent.
    """
    flood = [[mention(str(i)) for i in range(100)] for _ in range(20)]
    client = SpendingClient(flood)
    bot = MentionBot(client, FakeIndex(), daily_spend_cap_usd=0.05,
                     state_path=tmp_path / "s.json")
    for _ in range(20):
        await bot.tick("2026-08-26")
    assert client.spent_usd <= 0.15, (
        f"spent ${client.spent_usd:.3f} against a $0.05 cap")


@pytest.mark.anyio
async def test_the_cap_covers_replies_too(tmp_path):
    client = SpendingClient([[mention("0")]]
                            + [[mention(str(i))] for i in range(1, 40)])
    bot = MentionBot(client, FakeIndex(), daily_spend_cap_usd=0.5,
                     daily_reply_cap=1000, state_path=tmp_path / "s.json")
    for _ in range(40):
        await bot.tick("2026-08-26")
    assert client.spent_usd <= 0.8, f"spent ${client.spent_usd:.3f}"


@pytest.mark.anyio
async def test_the_ceiling_resets_the_next_day(tmp_path):
    """A cap is a rate, not a lifetime budget."""
    client = SpendingClient([[mention("0")], [mention("1")],
                             [mention("2")], [mention("3")]])
    # Below the cost of a single reply, so one is enough to close the day.
    bot = MentionBot(client, FakeIndex(), daily_spend_cap_usd=0.01,
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")                     # cold start
    assert await bot.tick("2026-08-26") == 1         # one reply, over the cap
    assert await bot.tick("2026-08-26") == 0, "idle for the rest of the day"

    assert await bot.tick("2026-08-27") == 1, "and working again tomorrow"


@pytest.mark.anyio
async def test_spend_is_recorded_even_when_a_cycle_returns_early(tmp_path):
    """The cold start reads, pays, and returns without replying. If that
    path did not record, the ceiling would never see the cost of a restart
    loop — and Render's disk is ephemeral, so restarts clear the state."""
    client = SpendingClient([[mention(str(i)) for i in range(30)]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    assert await bot.tick("2026-08-26") == 0
    assert bot.state.spent_today_usd == pytest.approx(0.030, abs=1e-6)


@pytest.mark.anyio
async def test_a_zero_cap_disables_the_ceiling(tmp_path):
    client = SpendingClient([[mention("0")], [mention("1")]])
    bot = MentionBot(client, FakeIndex(), daily_spend_cap_usd=0,
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 1


# --- answering only badged accounts ----------------------------------------

@pytest.mark.anyio
async def test_verified_only_skips_unbadged_accounts_for_free(tmp_path):
    """Skipped before retrieval, so an ignored account costs nothing beyond
    the read that already happened — no embedding, no model call, no reply.
    With links on, each skip is $0.209 not spent."""
    client = FakeClient([[mention("1")],
                         [mention("2", author="nobody", verified=False)]])
    index = FakeIndex()
    bot = MentionBot(client, index, verified_only=True,
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 0
    assert index.asked == [], "retrieval must not have been called"
    assert client.posted == []


@pytest.mark.anyio
async def test_verified_accounts_are_answered(tmp_path):
    client = FakeClient([[mention("1")],
                         [mention("2", author="ansem", verified=True)]])
    bot = MentionBot(client, FakeIndex(), verified_only=True,
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 1


@pytest.mark.anyio
async def test_the_filter_is_off_by_default(tmp_path):
    """A tool whose pitch is being useful to whoever asks should not require
    a paid checkmark by default."""
    client = FakeClient([[mention("1")],
                         [mention("2", author="nobody", verified=False)]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 1


def test_author_verification_is_parsed_from_the_expansion():
    """It arrives in includes.users, not on the post itself."""
    import json as _json

    body = _json.loads(_json.dumps({
        "data": [{"id": "1", "text": "@bot hi", "author_id": "u1",
                  "conversation_id": "1"}],
        "includes": {"users": [{"id": "u1", "verified": True,
                                "verified_type": "blue"}]},
    }))
    authors = {u["id"]: u for u in body["includes"]["users"]}
    m = body["data"][0]
    assert authors[m["author_id"]]["verified"] is True
    assert authors[m["author_id"]]["verified_type"] == "blue"


@pytest.mark.anyio
async def test_links_enabled_actually_posts_the_link(tmp_path):
    """The bug the first live reply hit.

    reply() asserted no-URL unconditionally, so switching links on produced
    a reply the client then refused to send — the guard rejecting the very
    mode that had been deliberately enabled. Nothing posted; it raised.
    """
    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://x.com/MarketBubble/status/2075316750439338088"

    client = FakeClient([[mention("1")], [mention("2")]])
    bot = MentionBot(client, FakeIndex(answer="Around 1:39:33 he bought it.",
                                       hits=[Hit()]),
                     include_links=True, state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 1
    assert Hit.deep_link in client.posted[0][1]


@pytest.mark.anyio
async def test_an_accidental_url_is_still_refused_when_links_are_off(tmp_path):
    """The guard must keep working in the default mode: a URL read aloud in
    a transcript, or one the model writes unprompted, still costs 13x."""
    client = FakeClient([[mention("1")], [mention("2")]])
    bot = MentionBot(client, FakeIndex(), include_links=False,
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    await bot.tick("2026-08-26")
    for _, text in client.posted:
        assert_linkless(text)


# --- a failure must not silently swallow the question ----------------------

class BreakingClient(FakeClient):
    """Raises on reply, like a bad request or a provider blip would."""

    def __init__(self, batches, fail_times=99):
        super().__init__(batches)
        self.fail_times = fail_times
        self.attempts = 0

    async def reply(self, text, to_post_id, allow_link=False):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError("posting failed")
        return await super().reply(text, to_post_id, allow_link)


@pytest.mark.anyio
async def test_a_failed_reply_does_not_lose_the_question(tmp_path):
    """The bug that ate the first live reply.

    since_id was advanced at the top of the loop, before the reply was
    attempted, so a crash mid-reply still marked the mention as seen. The
    question was never looked at again and nothing said so — the one
    outcome this bot cannot have, since being unable to find something is
    the complaint the whole project exists to answer.
    """
    client = BreakingClient([[mention("1")], [mention("2")], [mention("2")]],
                            fail_times=1)
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")                  # cold start
    assert await bot.tick("2026-08-26") == 0      # the reply raises
    assert bot.state.since_id == "1", "must not step past an unanswered one"
    assert await bot.tick("2026-08-26") == 1, "and the retry answers it"


@pytest.mark.anyio
async def test_a_permanently_broken_mention_is_eventually_stepped_over(tmp_path):
    """The other direction: one poison mention must not block the queue."""
    from app.x_bot import MAX_ATTEMPTS

    batches = [[mention("1")]] + [[mention("2")] for _ in range(MAX_ATTEMPTS)]
    client = BreakingClient(batches)
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    for _ in range(MAX_ATTEMPTS):
        await bot.tick("2026-08-26")
    assert bot.state.since_id == "2", "gives up rather than blocking forever"
    assert "2" not in bot.state.attempts, "and stops tracking it"


@pytest.mark.anyio
async def test_a_skipped_mention_still_advances(tmp_path):
    """Deliberate skips are handled, not failures — the queue must move.

    No highlight pool here, so "gm" is genuinely skipped rather than
    answered with a fact; the point is that the one behind it still gets
    through and since_id lands past both.
    """
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot gm"), mention("3")]])
    bot = MentionBot(client, FakeIndex(), highlights=[],
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 1
    assert bot.state.since_id == "3"


def test_link_mode_never_shows_two_different_timestamps():
    """Same contradiction as the no-link path, which was fixed there first.

    A reply read "...positioning it as a major entertainment IP. 1:39:33"
    above "Full episode (1:39:15):" — the model's own citation against the
    passage start, minutes apart, in the one detail this tool claims to get
    right.
    """
    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://x.com/MarketBubble/status/2075316750439338088"

    answered = "Luca bought Pudgy Penguins for 750 ETH, around 1:39:33."
    reply = format_reply(answered, [Hit()], include_links=True)
    assert "1:39:33" in reply, "the model's own citation survives"
    assert "1:39:15" not in reply, "the passage start must not compete"
    assert Hit.deep_link in reply


def test_link_mode_supplies_a_timestamp_when_the_answer_has_none():
    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://www.youtube.com/watch?v=abc&t=5955s"

    reply = format_reply("Luca bought Pudgy Penguins for 750 ETH.", [Hit()],
                         include_links=True)
    assert "1:39:15" in reply, "a reply with a link and no moment is useless"


def test_a_reply_with_a_link_fits_as_x_counts_it():
    from app.x_bot import weighted_length

    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://www.youtube.com/watch?v=" + "x" * 200

    reply = format_reply("Ansem said a great deal about this. " * 20, [Hit()],
                         include_links=True)
    assert weighted_length(reply) <= 280


def test_the_style_scales_with_what_the_account_can_post():
    """At 280 the instruction is "be short or you get cut off"; with real
    headroom it is "use the room, quote what was said". Asking for two
    terse sentences when 4000 characters are available wastes the account's
    only advantage."""
    from app.x_bot import reply_style

    short, long = reply_style(280), reply_style(4000)
    assert "complete short answer beats a truncated full one" in short
    assert "room for real detail" in long
    assert "201 characters" in short and "2880 characters" in long


def test_a_longer_limit_produces_a_longer_reply():
    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://x.com/MarketBubble/status/2075316750439338088"

    answer = "Luca bought Pudgy Penguins for 750 ETH. " * 40
    short = format_reply(answer, [Hit()], include_links=True, limit=280)
    long = format_reply(answer, [Hit()], include_links=True, limit=4000)
    assert len(long) > len(short) * 3
    assert Hit.deep_link in short and Hit.deep_link in long


def test_the_link_lands_where_the_answer_says_it_does():
    """The link is built from the top passage; the answer cites the line it
    actually used, often minutes away. Sending someone to the passage start
    while the text says 1:07:24 is the same broken promise as citing the
    wrong time."""
    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://www.youtube.com/watch?v=abc&t=5955s"

    reply = format_reply("Around 1:07:24 he explains the airdrop.", [Hit()],
                         include_links=True, limit=1500)
    assert "t=4044s" in reply, "1:07:24 is 4044 seconds"
    assert "t=5955s" not in reply
    assert "Jump to 1:07:24:" in reply


def test_an_x_link_is_never_given_a_timestamp_parameter():
    """X has no timestamp parameter for video and ignores one, so a link
    that carries it looks jumpable and is not."""
    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://x.com/MarketBubble/status/2075316750439338088"

    reply = format_reply("Around 1:07:24 he explains the airdrop.", [Hit()],
                         include_links=True, limit=1500)
    assert "t=" not in reply
    assert "Full episode:" in reply
    assert "1:07:24" in reply
    assert Hit.deep_link in reply


def test_the_moment_is_on_its_own_line():
    """Buried mid-sentence, the one thing this tool does was the least
    visible part of the reply."""
    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://www.youtube.com/watch?v=abc&t=5955s"

    reply = format_reply("He bought it for 750 ETH.", [Hit()],
                         include_links=True, limit=1500)
    lines = [ln for ln in reply.splitlines() if ln.strip()]
    assert lines[-2].startswith("Jump to 1:39:15")
    assert lines[-1] == Hit.deep_link


@pytest.mark.parametrize("stamp,seconds", [
    ("7:02", 422), ("1:07:24", 4044), ("4:01:47", 14507), ("0:30", 30),
])
def test_timestamp_parsing(stamp, seconds):
    from app.x_bot import _seconds

    assert _seconds(stamp) == seconds


def test_an_x_link_does_not_repeat_a_moment_the_answer_gave():
    """The first long reply said "Around 1:41:22 in the episode." and then
    "The bit above is at 1:41:22 —" underneath: the same fact twice, and a
    dangling dash where the card swallowed the URL."""
    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://x.com/MarketBubble/status/2075316750439338088"

    reply = format_reply("He bought it for 750 ETH. Around 1:41:22.", [Hit()],
                         include_links=True, limit=1500)
    assert reply.count("1:41:22") == 1, "stated once, not twice"
    assert not reply.rstrip().endswith("—")
    assert "Full episode:" in reply


def test_an_x_link_supplies_the_moment_when_the_answer_did_not():
    class Hit:
        title = "Market Bubble Ep 10"
        timestamp = "1:39:15"
        deep_link = "https://x.com/MarketBubble/status/2075316750439338088"

    reply = format_reply("He bought it for 750 ETH.", [Hit()],
                         include_links=True, limit=1500)
    assert "the moment is at 1:39:15" in reply


# --- opting out -------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "@bot stop", "@bot STOP", "@bot unsubscribe", "@bot opt out", "@bot opt-out",
    "@bot leave me alone", "@bot stop replying to me",
    "@bot don't reply to me again", "@bot no more replies", "@bot remove me",
])
def test_opt_out_is_recognised_generously(text):
    """X requires "a clear and easy way to opt out". Someone asking to be
    left alone should not have to guess the magic phrase."""
    from app.x_bot import asks_to_be_left_alone

    assert asks_to_be_left_alone(text)


@pytest.mark.parametrize("text", [
    "@bot what did ansem say about the stop loss",   # the false positive
    "@bot did they talk about a stopgap",
    "@bot when did ansem stop trading eth",
    "@bot what did tjr say",
])
def test_ordinary_questions_are_not_opt_outs(text):
    from app.x_bot import asks_to_be_left_alone

    assert not asks_to_be_left_alone(text)


@pytest.mark.anyio
async def test_an_opt_out_is_honoured_and_never_forgotten(tmp_path):
    """The promise in an opt-out is permanence. Honouring it for a while and
    then forgetting is worse than never having offered one."""
    path = tmp_path / "s.json"
    client = FakeClient([
        [mention("1")],
        [mention("2", text="@bot stop", author="tired")],
        [mention("3", text="@bot what did ansem say", author="tired")],
    ])
    index = FakeIndex()
    bot = MentionBot(client, index, state_path=path)
    await bot.tick("2026-08-26")                      # cold start
    assert await bot.tick("2026-08-26") == 0          # the opt-out itself
    assert "tired" in bot.state.opted_out
    assert await bot.tick("2026-08-26") == 0, "must not answer them again"
    assert index.asked == [], "and must not even ask the index"

    # Survives a restart, which is where "permanent" is actually tested.
    resumed = MentionBot(FakeClient([[mention("4", author="tired")]]),
                         FakeIndex(), state_path=path)
    assert await resumed.tick("2026-08-26") == 0


@pytest.mark.anyio
async def test_opting_out_does_not_silence_anyone_else(tmp_path):
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot stop", author="tired"),
                          mention("3", author="someone_else")]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 1
    assert client.posted[0][0] == "3"


# --- not swearing at strangers ---------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("they just fucked up the tokenomics", "they just f***ed up the tokenomics"),
    ("That shit is so fire", "That s*** is so fire"),
    ("he called it a shitcoin", "he called it a s***coin"),
    ("fuck", "f***"),
])
def test_profanity_is_masked_not_dropped(raw, expected):
    """X's rules: do not reply to users with potentially sensitive content,
    including profanity, unless they have indicated they want it. Someone
    asking what Ansem said about Ethereum has indicated no such thing.

    Masked rather than removed, so the quote stays faithful — the reader can
    see what was said without this account being the one that said it.
    """
    from app.x_bot import soften

    assert soften(raw) == expected


@pytest.mark.parametrize("clean", [
    "he shifted his position on ethereum",
    "the ticker was $CLAW and it ran 4x",
    "around 1:39:15 he explains the airdrop",
    "Scunthorpe",                       # the classic false positive
])
def test_ordinary_words_are_left_alone(clean):
    from app.x_bot import soften

    assert soften(clean) == clean


def test_a_quoted_transcript_line_reaches_the_reply_clean():
    """The real case: an answer quoting Ansem put the word in a reply to a
    stranger."""
    class Hit:
        title = "Market Bubble #4"
        timestamp = "26:56"
        deep_link = "https://www.youtube.com/watch?v=abc&t=1616s"

    answer = ('Around 26:56 Ansem argues "they just fucked up" the tokenomics.')
    reply = format_reply(answer, [Hit()], include_links=True, limit=1500)
    assert "fucked" not in reply
    assert "f***ed" in reply
    assert "26:56" in reply, "the citation still survives"


# --- paying for a link only when it does something -------------------------

YT = "https://www.youtube.com/watch?v=abc&t=422s"
XL = "https://x.com/MarketBubble/status/2075316750439338088"


@pytest.mark.parametrize("mode,link,expected", [
    ("always",   YT, True),
    ("always",   XL, True),
    ("seekable", YT, True),
    ("seekable", XL, False),      # opens at 0:00 — not worth $0.200
    ("off",      YT, False),
    ("off",      XL, False),
])
def test_link_modes(mode, link, expected):
    """A reply with a URL costs $0.200 against $0.015 whatever it points at,
    but only a YouTube link lands on the moment. An X broadcast link opens a
    four-hour video at 0:00, which the timestamp in the text already does
    for a fraction of the price."""
    from app.x_bot import wants_link

    assert wants_link(mode, link) is expected


def test_seekable_mode_links_a_youtube_answer():
    class Hit:
        title = "TJR On Why Attention Beat Money"
        timestamp = "7:02"
        deep_link = YT

    reply = format_reply("Around 7:02 TJR called attention the best currency.",
                         [Hit()], include_links="seekable", limit=1500)
    assert YT in reply
    assert "Jump to 7:02:" in reply


def test_seekable_mode_skips_an_x_broadcast_link():
    class Hit:
        title = "LIVE W/ LUCA NETZ: Market Bubble Ep 10"
        timestamp = "1:41:22"
        deep_link = XL

    reply = format_reply("Around 1:41:22 Luca explained the airdrop.",
                         [Hit()], include_links="seekable", limit=1500)
    assert "http" not in reply, "no link, so no $0.200 charge"
    assert "1:41:22" in reply, "but the moment is still named"
    assert "Market Bubble Ep 10" in reply, "and so is the episode"


def test_the_old_booleans_still_mean_what_they_meant():
    """render.yaml and .env carried true/false before the third mode
    existed, and a deploy that read those as neither would silently stop
    linking or start linking on every reply."""
    from app.x_bot import wants_link

    class Hit:
        title = "Ep 10"
        timestamp = "1:00:00"
        deep_link = XL

    assert XL in format_reply("x", [Hit()], include_links=True, limit=1500)
    assert "http" not in format_reply("x", [Hit()], include_links=False,
                                      limit=1500)
    assert wants_link("always", XL) and not wants_link("off", XL)


@pytest.mark.parametrize("raw,expected", [
    ("he said it [2:29:34]", "he said it 2:29:34"),
    ("a range [2:29:34–2:33:04] leaked", "a range 2:29:34–2:33:04 leaked"),
    ("[2:29:34-2:33:04] hyphen", "2:29:34–2:33:04 hyphen"),
    ("around [7:02] and [1:39:15]", "around 7:02 and 1:39:15"),
])
def test_bracketed_timestamp_ranges_are_cleaned(raw, expected):
    """A live reply went out reading "[2:29:34–2:33:04]".

    The pattern only knew about a single timestamp, and the model writes
    ranges whenever an answer spans a stretch of conversation — which a
    1500-character reply does constantly, so the longer replies made this
    far more likely than the short ones ever did.
    """
    from app.x_bot import plain_text

    assert plain_text(raw) == expected


# --- summarise episode N ---------------------------------------------------

@pytest.mark.parametrize("asked,number", [
    ("summarize episode 14", 14),
    ("summarise ep 12", 12),
    ("summary of episode 3", 3),
    ("recap #9", 9),
    ("whats the rundown on episode 16", 16),
    ("what happened in episode 1", 1),
])
def test_summary_requests_are_recognised(asked, number):
    from app.x_bot import summary_request

    assert summary_request(asked) == number


@pytest.mark.parametrize("asked", [
    "what did ansem say about eth",
    "summarise the sec stuff",          # no episode number
    "what did tjr say in the episode",
])
def test_ordinary_questions_are_not_summary_requests(asked):
    from app.x_bot import summary_request

    assert summary_request(asked) is None


@pytest.mark.parametrize("title,number", [
    ("The Best Day Crypto Has Had In Months | Market Bubble #16", 16),
    ("LIVE W/ TJR & Mert Market Bubble EP 8 - Presented by", 8),
    ("Our full conversation with Orangie.", None),
])
def test_episode_numbers_are_read_from_titles(title, number):
    from app.x_bot import episode_number

    assert episode_number(title) == number


class FakeSummaries:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    async def list_all(self):
        self.calls += 1
        return self.rows


@pytest.mark.anyio
async def test_a_summary_request_is_answered_from_storage(tmp_path):
    """No retrieval and no model call: the summary already exists, already
    carries timestamps, and cannot come back different from the one on the
    website."""
    rows = [{"title": "Market Bubble #14", "summary":
             "**TL;DR** — Tushar Jain on Solana and Zcash, around 1:15:37."}]
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot summarize episode 14")]])
    index = FakeIndex()
    bot = MentionBot(client, index, summaries=FakeSummaries(rows),
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-27")
    assert await bot.tick("2026-08-27") == 1
    assert index.asked == [], "must not have gone to retrieval"

    reply = client.posted[0][1]
    assert "Tushar Jain" in reply and "1:15:37" in reply
    assert "**" not in reply, "markdown does not render on X"
    assert "http" not in reply, "a summary carries no link"


@pytest.mark.anyio
async def test_the_summary_list_is_fetched_once(tmp_path):
    rows = [{"title": "Market Bubble #14", "summary": "x " * 60}]
    store = FakeSummaries(rows)
    client = FakeClient([[mention("1")]]
                        + [[mention(str(i), text="@bot recap #14")]
                           for i in range(2, 6)])
    bot = MentionBot(client, FakeIndex(), summaries=store,
                     state_path=tmp_path / "s.json")
    for _ in range(5):
        await bot.tick("2026-08-27")
    assert store.calls == 1, "32 summaries change only when an episode lands"


@pytest.mark.anyio
async def test_an_unknown_episode_says_so(tmp_path):
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot summarize episode 99")]])
    bot = MentionBot(client, FakeIndex(), summaries=FakeSummaries([]),
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-27")
    assert await bot.tick("2026-08-27") == 1
    assert client.posted[0][1] == "I don't have episode 99 indexed."


@pytest.mark.anyio
async def test_the_longer_cut_wins_when_a_show_exists_twice(tmp_path):
    """A show is often both a YouTube cut and a live broadcast. The summary
    of a cut is a summary of a cut."""
    rows = [{"title": "Market Bubble #10", "summary": "short one"},
            {"title": "LIVE W/ LUCA NETZ: Market Bubble Ep 10",
             "summary": "the full broadcast, considerably longer " * 8}]
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot summarise ep 10")]])
    bot = MentionBot(client, FakeIndex(), summaries=FakeSummaries(rows),
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-27")
    await bot.tick("2026-08-27")
    assert "full broadcast" in client.posted[0][1]


def test_a_summary_keeps_its_paragraphs():
    """A 3,000-character summary flattened into one block is a wall nobody
    reads, and the structure — TL;DR, then a timestamped topic list — is
    most of what makes it legible.

    strip_urls used text.split(), which splits on every kind of whitespace
    and rejoins with spaces. Harmless while replies were two sentences;
    it destroyed the summary.
    """
    from app.x_bot import format_summary

    raw = ("**TL;DR** — the episode in one paragraph.\n"
           "\n"
           "Topics\n"
           "0:00:00 intro and recap\n"
           "0:53:10 guest interview begins\n")
    out = format_summary(raw, "Market Bubble #14", 4000)
    assert out.count("\n") >= 4, "paragraph structure must survive"
    assert "Topics" in out
    assert "0:53:10" in out
    assert "**" not in out


def test_short_answers_still_get_their_newlines_collapsed():
    """The default stays as it was: a two-sentence answer arriving with
    stray newlines reads as broken."""
    from app.x_bot import plain_text

    assert plain_text("he said\nit\nhere") == "he said it here"


# --- a deploy must not eat a question --------------------------------------

def _iso(minutes_ago: float) -> str:
    from datetime import UTC, datetime, timedelta
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()


def test_recency_window():
    from app.x_bot import _is_recent

    assert _is_recent(_iso(2))
    assert _is_recent(_iso(29))
    assert not _is_recent(_iso(45))
    assert not _is_recent(""), "unknown age is treated as old"
    assert not _is_recent("not a date")


@pytest.mark.anyio
async def test_a_cold_start_still_answers_a_recent_question(tmp_path):
    """The failure that made the account look broken.

    Render's disk is ephemeral, so every deploy hands the bot an empty state
    file and it cold-starts. Skipping everything meant five deploys in an
    hour ate the same question twice while someone was watching it not
    reply.
    """
    old = Mention(id="100", text="@bot what did ansem say", author_id="a",
                  conversation_id="100", created_at=_iso(600))
    fresh = Mention(id="200", text="@bot what did tjr say", author_id="b",
                    conversation_id="200", created_at=_iso(3))
    client = FakeClient([[old, fresh], [old, fresh]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")

    assert await bot.tick("2026-08-27") == 1, "the fresh one is answered"
    assert client.posted[0][0] == "200"


@pytest.mark.anyio
async def test_a_cold_start_still_skips_a_backlog(tmp_path):
    """The protection that behaviour exists for: replying to a month of old
    mentions in one burst is how accounts get suspended."""
    old = [Mention(id=str(i), text="@bot what did ansem say", author_id="a",
                   conversation_id=str(i), created_at=_iso(60 * 24 * i))
           for i in range(1, 12)]
    client = FakeClient([old])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")

    assert await bot.tick("2026-08-27") == 0
    assert client.posted == []
    assert bot.state.since_id == "11"


@pytest.mark.anyio
async def test_a_cold_start_answers_only_the_recent_half(tmp_path):
    old = Mention(id="100", text="@bot what did ansem say", author_id="a",
                  conversation_id="100", created_at=_iso(900))
    fresh = Mention(id="200", text="@bot what did tjr say", author_id="b",
                    conversation_id="200", created_at=_iso(5))
    client = FakeClient([[old, fresh], [old, fresh]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    await bot.tick("2026-08-27")
    assert [p for p, _ in client.posted] == ["200"], "not the old one"


def test_topics_get_their_own_blocks():
    """Run together, a dozen three-line entries are a wall with nowhere for
    the eye to land — every timestamp buried mid-paragraph, reading as part
    of the sentence before it. The blank line makes each one an anchor you
    can scan down, which is the only way anyone finds the bit they came
    for."""
    from app.x_bot import format_summary

    raw = ("**TL;DR** — the episode in one paragraph.\n"
           "\n"
           "Topics\n"
           "0:00:00 Show open and announcements\n"
           "0:07:15 A fund forced to liquidate\n"
           "0:18:19 A trader who turned 500 into 40M\n")
    out = format_summary(raw, "Market Bubble #13", 4000)

    assert "0:00:00 · Show open" in out, "separator after the time"
    assert "\n\n0:07:15" in out, "a blank line before each entry"
    assert "\n\n0:18:19" in out
    assert "**" not in out


def test_prose_paragraphs_are_left_alone():
    """Only timestamped lines are topic entries. A paragraph that happens to
    mention a time mid-sentence is prose."""
    from app.x_bot import format_summary

    raw = ("TL;DR — around 1:39:15 he explains the airdrop, and the rest of "
           "the paragraph continues normally.\n"
           "\n"
           "Topics\n"
           "0:00:00 intro\n")
    out = format_summary(raw, "Ep 10", 4000)
    assert "around 1:39:15 he explains" in out, "prose is untouched"
    assert "0:00:00 · intro" in out


def test_a_summary_carries_the_episode_link():
    """Worth its $0.200 here in a way it is not on a two-sentence answer: a
    summary is what someone reads while deciding whether to watch the
    episode, so the thing to hand them next is the episode."""
    from app.x_bot import format_summary

    url = "https://www.youtube.com/watch?v=47AACkIhtG8"
    out = format_summary("TL;DR — the episode.\n\n0:00:00 intro\n",
                         "Market Bubble #13", 4000, url=url)
    assert out.endswith(url)
    assert "Full episode:" in out
    assert "Market Bubble #13" not in out, "the card already shows the title"


def test_a_summary_without_a_link_keeps_its_title():
    from app.x_bot import format_summary

    out = format_summary("TL;DR — the episode.\n", "Market Bubble #13", 4000)
    assert "Market Bubble #13" in out
    assert "http" not in out


@pytest.mark.anyio
async def test_links_off_means_no_link_on_summaries_either(tmp_path):
    rows = [{"title": "Market Bubble #14", "summary": "TL;DR — a summary.",
             "url": "https://www.youtube.com/watch?v=abc"}]
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot summarize episode 14")]])
    bot = MentionBot(client, FakeIndex(), summaries=FakeSummaries(rows),
                     include_links=False, state_path=tmp_path / "s.json")
    await bot.tick("2026-08-27")
    await bot.tick("2026-08-27")
    assert "http" not in client.posted[0][1]


@pytest.mark.anyio
async def test_links_on_means_the_summary_gets_one(tmp_path):
    url = "https://www.youtube.com/watch?v=abc"
    rows = [{"title": "Market Bubble #14", "summary": "TL;DR — a summary.",
             "url": url}]
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot summarize episode 14")]])
    bot = MentionBot(client, FakeIndex(), summaries=FakeSummaries(rows),
                     include_links="always", state_path=tmp_path / "s.json")
    await bot.tick("2026-08-27")
    await bot.tick("2026-08-27")
    assert url in client.posted[0][1]


@pytest.mark.parametrize("raw", [
    "**TL;DR** — the episode in a paragraph.",
    "TL;DR — the episode in a paragraph.",
    "TL;DR: the episode in a paragraph.",
    "tldr - the episode in a paragraph.",
])
def test_the_tldr_label_is_dropped(raw):
    """It is the first thing anyone sees in a summary reply, and it spends
    characters telling them what they already know — they asked for a
    summary. The website keeps it, where a labelled block helps someone
    scanning down a page."""
    from app.x_bot import format_summary

    out = format_summary(raw, "Ep 16", 4000)
    assert out.startswith("the episode in a paragraph.")
    assert "TL" not in out.upper()[:6]


def test_a_summary_without_the_label_is_untouched():
    from app.x_bot import format_summary

    out = format_summary("This episode covers a green day.", "Ep 16", 4000)
    assert out.startswith("This episode covers a green day.")


def test_polling_is_frequent_but_not_instant():
    """Nearly all the latency was here: answering takes 4-8 seconds, and at
    a 60s interval a mention waited up to 84 just to be noticed.

    Polling costs nothing — X charges per resource returned and dedupes
    within the UTC day — so the only reason not to go lower is that a reply
    three seconds after the question reads as a machine, and "reply speed no
    human could achieve" is a documented suspension trigger.
    """
    pauses = [MentionBot.pause_seconds(20.0) for _ in range(500)]
    assert min(pauses) >= 10, "not so fast it looks automated"
    assert max(pauses) <= 30, "not so slow the question sits unseen"
    assert len(set(round(p, 3) for p in pauses)) > 400, "still jittered"


# --- never answer the same mention twice, even across a restart ------------

class RestartingClient(FakeClient):
    """Knows what it has already posted, the way X does."""

    def __init__(self, batches):
        super().__init__(batches)
        self._answered: set[str] = set()

    async def replied_to(self, limit=100):
        return set(self._answered)

    async def reply(self, text, to_post_id, allow_link=False):
        self._answered.add(to_post_id)
        return await super().reply(text, to_post_id, allow_link)


@pytest.mark.anyio
async def test_a_redeploy_does_not_answer_the_same_question_again(tmp_path):
    """The bug that put three replies under one question.

    The replied-set lives in the state file that a deploy wipes, so
    "answer anything recent" meant re-answering what had already been
    answered — once per deploy, and there were three.
    """
    fresh = Mention(id="500", text="@bot what did tjr say", author_id="a",
                    conversation_id="500", created_at=_iso(4))

    first = RestartingClient([[fresh], [fresh]])
    bot = MentionBot(first, FakeIndex(), state_path=tmp_path / "a.json")
    assert await bot.tick("2026-08-27") == 1

    # Redeploy: same account, brand new state file.
    second = RestartingClient([[fresh]])
    second._answered = set(first._answered)
    restarted = MentionBot(second, FakeIndex(), state_path=tmp_path / "b.json")
    assert await restarted.tick("2026-08-27") == 0, "already answered"
    assert second.posted == []


@pytest.mark.anyio
async def test_a_redeploy_still_answers_a_question_it_has_not_seen(tmp_path):
    answered = Mention(id="500", text="@bot what did tjr say", author_id="a",
                       conversation_id="500", created_at=_iso(4))
    new = Mention(id="600", text="@bot what did ansem say", author_id="b",
                  conversation_id="600", created_at=_iso(2))
    client = RestartingClient([[answered, new], [answered, new]])
    client._answered = {"500"}
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    assert await bot.tick("2026-08-27") == 1
    assert [p for p, _ in client.posted] == ["600"]


@pytest.mark.anyio
async def test_if_x_cannot_be_read_the_backlog_is_skipped(tmp_path):
    """Failing safe: without the record, repeating itself in public is worse
    than missing a question."""
    class Broken(FakeClient):
        async def replied_to(self, limit=100):
            raise RuntimeError("timeout")

    fresh = Mention(id="500", text="@bot what did tjr say", author_id="a",
                    conversation_id="500", created_at=_iso(3))
    client = Broken([[fresh]])
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    assert await bot.tick("2026-08-27") == 0
    assert client.posted == []


# --- the fixed replies must not be byte-identical --------------------------

def test_the_contract_reply_varies_between_askers():
    """X forbids "duplicative or substantially similar posts on one
    account". A generated answer differs every time; the contract address
    was byte-identical however many people asked, and twenty identical posts
    is the shape that rule describes."""
    from app.x_bot import pinned_answer

    asks = ["whats the ca", "ca pls", "contract address?", "ca?",
            "can i get the CA", "drop the mint", "whats the contract"]
    replies = {pinned_answer(a, CA, "MarketBubbleSearch") for a in asks}
    assert len(replies) >= 3, "several askers should not get one phrasing"
    for r in replies:
        assert CA in r, "every phrasing still carries the address"


def test_the_same_asker_gets_a_stable_answer():
    """Varied by hashing the question, not at random: someone asking twice
    should not think they got two different addresses."""
    from app.x_bot import pinned_answer

    first = pinned_answer("whats the ca", CA, "MBS")
    assert first == pinned_answer("whats the ca", CA, "MBS")


def test_a_hostile_address_is_never_echoed_in_any_phrasing():
    from app.x_bot import pinned_answer

    hostile = "is the ca 7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
    got = pinned_answer(hostile, CA, "MBS")
    assert CA in got and "7xKXtg" not in got


def test_the_miss_reply_varies_too():
    """Same rule: "I couldn't find that" was identical every time, and a
    miss is one of the commonest replies this bot makes."""
    from app.podcast import NOT_FOUND_ANSWER

    answers = [f"{NOT_FOUND_ANSWER}. Nothing about {topic} in the excerpts."
               for topic in ("taylor swift", "peru", "cake", "the weather")]
    replies = {format_reply(a, [FakeHit()]) for a in answers}
    assert len(replies) >= 2, "misses should not all read identically"
    for r in replies:
        assert "couldn" in r.lower(), "and all still say it plainly"
        assert "·" not in r, "still no citation on a miss"


def test_a_reply_is_costed_at_what_x_actually_charges():
    """X publishes $0.015 plain, $0.200 with a URL, $0.010 summoned. Every
    reply here carries a URL in the text field and is triggered by a
    mention, so on paper it should hit one of the other two — measured, it
    is charged the plain rate.

    Estimating at the premium made the daily ceiling stop the bot at a
    twelfth of the spend it was set to allow: $5 became 24 replies rather
    than 300.
    """
    from app.x_api import PRICE_POST, XClient, XCredentials

    client = XClient(XCredentials("k", "s", "t", "ts"), bot_user_id="1",
                     dry_run=True)
    assert PRICE_POST == 0.015
    assert client.spent_usd == 0.0


# --- a non-answer must never be posted, however it is worded ---------------

@pytest.mark.parametrize("answer", [
    # The one that went out live, under a real post, with a link attached.
    "I don't have enough information to answer this question. The excerpts "
    "provided don't contain a clear discussion of what \"beauty\" means, "
    "like around 1:31:26-1:33:04. Could you ask about a specific moment?",
    "I'm here to answer questions about the Market Bubble podcast using the "
    "excerpts I've been given.",
    "The excerpts don't mention that. Can you be more specific?",
    "Your question is pretty broad! Could you clarify what you mean?",
    "That is not discussed in the excerpts provided.",
])
def test_a_deflection_is_never_posted(answer):
    """Third time a model non-answer reached a reply, each in different
    words. Matching one canonical phrase kept failing, so this matches the
    shape instead: a stock refusal, or an answer that closes by asking the
    reader a question — which a real answer to "what did X say" does not do.
    """
    from app.x_bot import is_a_deflection

    assert is_a_deflection(answer)


@pytest.mark.parametrize("answer", [
    "Around 7:02 TJR called attention the best currency nowadays.",
    "Luca bought Pudgy Penguins for 750 ETH during NFT mania, at 1:41:22.",
    "I couldn't find that in the episodes I've indexed.",
    "Ansem argued Ethereum got outcompeted — around 26:56 — and that its "
    "tokenomics were mishandled.",
])
def test_real_answers_are_not_mistaken_for_deflections(answer):
    from app.x_bot import is_a_deflection

    assert not is_a_deflection(answer)


@pytest.mark.anyio
async def test_a_deflection_reaches_no_one(tmp_path):
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot you beauty")]])
    deflecting = FakeIndex(
        answer="I don't have enough information to answer this. The excerpts "
               "don't contain that, around 1:31:26. Could you ask about a "
               "specific moment?")
    # No pool, so the only possible reply is the deflection itself.
    bot = MentionBot(client, deflecting, highlights=[],
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-27")
    assert await bot.tick("2026-08-27") == 0
    assert client.posted == []


@pytest.mark.parametrize("text", [
    "you beauty", "absolute legend", "lets go", "no way", "holy",
])
def test_exclamations_are_not_questions(text):
    """"you beauty @mbubbleSearch" got a full rambling reply."""
    from app.x_bot import looks_like_a_question

    assert not looks_like_a_question(text)


# --- what happens when a lot of people tag it at once ----------------------

@pytest.mark.anyio
async def test_a_flood_of_mentions_does_not_crash_or_overspend(tmp_path):
    """The realistic bad day: the account gets noticed and two hundred
    people tag it inside an hour.

    Nothing here should throw, the caps should hold exactly, and the bot
    should still be answering the newest questions rather than stuck on the
    oldest.
    """
    flood = [Mention(id=str(1000 + i), text=f"@bot what did guest {i} say",
                     author_id=f"u{i}", conversation_id=str(1000 + i),
                     created_at=_iso(5))
             for i in range(200)]
    client = FakeClient([flood[:100], flood[100:]])
    bot = MentionBot(client, FakeIndex(), daily_reply_cap=50,
                     daily_spend_cap_usd=12.0, state_path=tmp_path / "s.json")

    total = 0
    for _ in range(6):
        total += await bot.tick("2026-08-27")

    assert total <= 50, f"reply cap breached: {total}"
    assert len(client.posted) == total
    assert len({p for p, _ in client.posted}) == total, "no duplicates"
    assert bot.state.spent_today_usd <= 12.0


@pytest.mark.anyio
async def test_one_broken_mention_does_not_block_the_queue(tmp_path):
    """A poison mention must not stop everyone behind it. It is retried a
    few times, then stepped over."""
    from app.x_bot import MAX_ATTEMPTS

    bad = Mention(id="1", text="@bot what did ansem say", author_id="a",
                  conversation_id="1", created_at=_iso(5))
    good = Mention(id="2", text="@bot what did tjr say", author_id="b",
                   conversation_id="2", created_at=_iso(4))

    class Poison(FakeClient):
        async def reply(self, text, to_post_id, allow_link=False):
            if to_post_id == "1":
                raise RuntimeError("this one always fails")
            return await super().reply(text, to_post_id, allow_link)

    client = Poison([[bad, good]] * (MAX_ATTEMPTS + 3))
    bot = MentionBot(client, FakeIndex(), state_path=tmp_path / "s.json")
    for _ in range(MAX_ATTEMPTS + 3):
        await bot.tick("2026-08-27")

    assert [p for p, _ in client.posted] == ["2"], "the good one got through"


@pytest.mark.anyio
async def test_the_same_person_asking_twenty_times_is_still_capped(tmp_path):
    """One account cannot drain the day on its own."""
    spam = [Mention(id=str(2000 + i), text=f"@bot what did ansem say {i}",
                    author_id="spammer", conversation_id=str(2000 + i),
                    created_at=_iso(3))
            for i in range(40)]
    client = FakeClient([spam, spam[20:]])
    bot = MentionBot(client, FakeIndex(), daily_reply_cap=10,
                     state_path=tmp_path / "s.json")
    total = sum([await bot.tick("2026-08-27") for _ in range(4)])
    assert total <= 10


# --- a compliment gets a fact, not silence ---------------------------------

POOL = [
    {"text": "Ansem said he made $1.37 million from a creator fee",
     "timestamp": "1:17:15", "title": "Market Bubble Ep 10"},
    {"text": "Luca sold Artifact to Nike for $2.5 million",
     "timestamp": "1:39:44", "title": "Market Bubble Ep 10"},
    {"text": "Kendrick Perkins turned $250,000 into roughly $2 million",
     "timestamp": "1:40:59", "title": "Market Bubble Ep 12"},
]


@pytest.mark.anyio
async def test_a_compliment_gets_a_fact(tmp_path):
    """Silence in front of someone who just said something nice is a wasted
    moment — they are looking at the account, and a demonstration convinces
    where a thank-you does not."""
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot very cool concept!")]])
    index = FakeIndex()
    bot = MentionBot(client, index, highlights=POOL,
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-27")
    assert await bot.tick("2026-08-27") == 1
    assert index.asked == [], "no retrieval, no model call"

    reply = client.posted[0][1]
    assert any(h["text"][:30] in reply for h in POOL), "carries a real fact"
    assert any(h["timestamp"] in reply for h in POOL), "and its moment"


@pytest.mark.anyio
async def test_the_same_fact_is_never_offered_twice(tmp_path):
    """Posting the same fact twice is the duplicative-content problem in a
    different costume."""
    batches = [[mention("1")]] + [[mention(str(i), text="@bot nice work")]
                                  for i in range(2, 5)]
    client = FakeClient(batches)
    bot = MentionBot(client, FakeIndex(), highlights=POOL,
                     state_path=tmp_path / "s.json")
    for _ in range(4):
        await bot.tick("2026-08-27")

    facts = [next(h["text"] for h in POOL if h["text"][:30] in body)
             for _, body in client.posted]
    assert len(facts) == len(set(facts)), f"repeated a fact: {facts}"


@pytest.mark.anyio
async def test_the_pool_reshuffles_once_it_is_spent(tmp_path):
    """Three facts and five compliments: it must keep answering rather than
    fall silent once every one has been used."""
    batches = [[mention("1")]] + [[mention(str(i), text="@bot lfg")]
                                  for i in range(2, 8)]
    client = FakeClient(batches)
    bot = MentionBot(client, FakeIndex(), highlights=POOL,
                     state_path=tmp_path / "s.json")
    for _ in range(7):
        await bot.tick("2026-08-27")
    assert len(client.posted) >= 5


@pytest.mark.anyio
async def test_no_pool_means_silence_not_a_crash(tmp_path):
    """An empty or missing pool must fall back to the old behaviour."""
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot very cool")]])
    bot = MentionBot(client, FakeIndex(), highlights=[],
                     state_path=tmp_path / "s.json")
    await bot.tick("2026-08-27")
    assert await bot.tick("2026-08-27") == 0
    assert client.posted == []


def test_a_highlight_reply_carries_no_link():
    """No link: a fact nobody asked for should not also cost the URL rate,
    and there is nothing to click through to mid-sentence."""
    from app.x_bot import format_highlight

    out = format_highlight(POOL[0], "seed")
    assert "http" not in out
    assert "1:17:15" in out


def test_highlight_openers_vary():
    from app.x_bot import format_highlight

    leads = {format_highlight(POOL[0], s).splitlines()[0]
             for s in ("a", "b", "c", "d", "e", "f", "g")}
    assert len(leads) >= 3, "twenty compliments should not open identically"
