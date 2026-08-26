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
    assert got.startswith("MarketBubbleSearch CA:"), (
        "a bare address read out of context says nothing about "
        "which token it belongs to")


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
    assert reply.startswith("MarketBubbleSearch CA:") and CA in reply
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
    """It must cost nothing — that is the point of gating before retrieval."""
    client = FakeClient([[mention("1")],
                         [mention("2", text="@bot very cool concept!")]])
    index = FakeIndex()
    bot = MentionBot(client, index, state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    assert await bot.tick("2026-08-26") == 0
    assert index.asked == [], "retrieval must not have been called"
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
    from app.x_bot import REPLY_STYLE

    client = FakeClient([[mention("1")], [mention("2")]])
    index = FakeIndex()
    bot = MentionBot(client, index, state_path=tmp_path / "s.json")
    await bot.tick("2026-08-26")
    await bot.tick("2026-08-26")
    assert index.instructed == REPLY_STYLE


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
        got = self._batches.pop(0) if self._batches else []
        self.spent_usd += len(got) * PRICE_OWNED_READ
        return got

    async def reply(self, text, to_post_id):
        self.spent_usd += 0.200          # a reply carrying a link
        return await super().reply(text, to_post_id)


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
    bot = MentionBot(client, FakeIndex(), daily_spend_cap_usd=0.05,
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
