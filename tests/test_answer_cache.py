"""The answer cache, and the two ways it could be worse than no cache.

Serving a fast wrong answer is worse than serving a slow right one, so most
of these are about what must NOT be shared: answers across surfaces, across
brief/verbose, or into a conversation that had prior turns.
"""

import time

import pytest

from app.answer_cache import AnswerCache, make_key, normalise
from app.schemas import ChatRequest

# --- keys ------------------------------------------------------------------

def test_trivial_variations_are_one_entry():
    """Casing, spacing and a trailing question mark are not different
    questions, and treating them as such would leave most repeats uncached."""
    a = make_key("What are the fees?", surface="clawpump")
    b = make_key("  what are   the fees ", surface="clawpump")
    assert a == b


def test_different_words_are_different_entries():
    assert (make_key("what are the fees", surface="clawpump")
            != make_key("what are the limits", surface="clawpump"))


def test_surfaces_never_share_an_answer():
    """The whole reason the key is not just the question.

    Two support bots on this service answer "what are the fees" from
    different documentation. A shared key would serve one product's answer
    to the other's users — confidently, and with the wrong sources attached.
    """
    assert (make_key("what are the fees", surface="clawpump")
            != make_key("what are the fees", surface="concierge"))


def test_brief_and_full_answers_are_separate():
    """A chat client asks for a short answer; the web page has room for a
    thorough one. Sharing the entry would give one of them the wrong shape."""
    assert (make_key("what are the fees", surface="concierge", brief=True)
            != make_key("what are the fees", surface="concierge", brief=False))


def test_top_k_is_part_of_the_key():
    assert (make_key("hyperliquid", surface="podcast", top_k=3)
            != make_key("hyperliquid", surface="podcast", top_k=8))


def test_normalise_keeps_internal_punctuation():
    # Only trailing punctuation is noise. "what's" and "whats" are different
    # strings and stripping inside a word would collide unrelated questions.
    assert normalise("What's the CA?") == "what's the ca"


# --- storage ---------------------------------------------------------------

def test_round_trip_and_counters():
    c = AnswerCache(max_entries=4, ttl_seconds=60)
    assert c.get("k") is None
    c.put("k", "answer")
    assert c.get("k") == "answer"
    assert c.state()["hits"] == 1
    assert c.state()["misses"] == 1


def test_expired_entries_are_not_served():
    c = AnswerCache(max_entries=4, ttl_seconds=0.01)
    c.put("k", "stale")
    time.sleep(0.05)
    assert c.get("k") is None


def test_evicts_least_recently_used_not_oldest_written():
    """`a` is written first but read most recently, so `b` is what goes.

    Getting this backwards would evict exactly the popular questions the
    cache exists to hold.
    """
    c = AnswerCache(max_entries=2, ttl_seconds=60)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")
    c.put("c", 3)
    assert c.get("a") == 1
    assert c.get("b") is None


def test_bounded_under_pressure():
    """A public endpoint means anyone can ask many distinct questions. The
    cache must cost bounded memory, not memory proportional to that."""
    c = AnswerCache(max_entries=10, ttl_seconds=60)
    for i in range(500):
        c.put(f"k{i}", i)
    assert c.state()["entries"] == 10


def test_clear_returns_count():
    c = AnswerCache(max_entries=10, ttl_seconds=60)
    c.put("a", 1)
    c.put("b", 2)
    assert c.clear() == 2
    assert c.get("a") is None


def test_zero_entries_disables_it():
    c = AnswerCache(max_entries=0, ttl_seconds=60)
    assert not c.enabled
    c.put("a", 1)
    assert c.get("a") is None


# --- request-level policy --------------------------------------------------

def test_only_single_turn_requests_are_cacheable():
    from app.main import _cacheable

    assert _cacheable(ChatRequest(message="what are the fees"))
    assert not _cacheable(ChatRequest(
        message="what about for perps?",
        history=[{"role": "user", "content": "what are the fees"},
                 {"role": "assistant", "content": "..."}],
    ))


@pytest.mark.anyio
async def test_second_identical_question_skips_the_model():
    """The point of the whole exercise: one model call, two answers."""
    from app import main
    from app.schemas import ChatResponse

    calls = {"n": 0}

    class FakeRetriever:
        async def search(self, query, filters=None, top_k=None, namespace=None):
            return []

    class FakeAgent:
        async def answer(self, message, history, chunks, brief=False):
            calls["n"] += 1
            return ChatResponse(answer="65%", sources=[], model="test")

    cache = AnswerCache(max_entries=8, ttl_seconds=60)
    body = ChatRequest(message="what are the fees")
    for _ in range(2):
        r = await main.clawpump_chat(body, retriever=FakeRetriever(),
                                     agent=FakeAgent(), answers=cache)
        assert r.answer == "65%"
    assert calls["n"] == 1


@pytest.mark.anyio
async def test_a_follow_up_is_never_answered_from_cache():
    """Two conversations can contain the same follow-up text and mean
    entirely different things."""
    from app import main
    from app.schemas import ChatResponse

    calls = {"n": 0}

    class FakeRetriever:
        async def search(self, query, filters=None, top_k=None, namespace=None):
            return []

    class FakeAgent:
        async def answer(self, message, history, chunks, brief=False):
            calls["n"] += 1
            return ChatResponse(answer=f"answer {calls['n']}", sources=[],
                                model="test")

    cache = AnswerCache(max_entries=8, ttl_seconds=60)
    body = ChatRequest(
        message="what about that?",
        history=[{"role": "user", "content": "fees"},
                 {"role": "assistant", "content": "65%"}],
    )
    await main.clawpump_chat(body, retriever=FakeRetriever(),
                             agent=FakeAgent(), answers=cache)
    await main.clawpump_chat(body, retriever=FakeRetriever(),
                             agent=FakeAgent(), answers=cache)
    assert calls["n"] == 2
