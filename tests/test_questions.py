"""The question log, and the two ways it must not fail.

It must never break a reply. A reply that was composed, checked and
approved must not be lost because Pinecone had a bad minute — the log is
a nice-to-have and the reply is the product.

And it must not lie about being empty. The first version of list_all
iterated the pages returned by index.list() directly, when the ids
actually live on page.vectors. Writes landed, reads came back empty, and
nothing raised: it would have reported "nothing logged yet" forever while
quietly recording every question. That is the failure this file exists
for, and it is the third one today with the same shape — something broken
that looks exactly like a normal, quiet, correct result.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.questions import QuestionLog  # noqa: E402


class FakeVector:
    def __init__(self, metadata):
        self.metadata = metadata


class FakePage:
    """What Pinecone actually returns: ids live on .vectors, not the page."""

    def __init__(self, ids):
        self.vectors = [type("Id", (), {"id": i})() for i in ids]


class FakeIndex:
    def __init__(self, explode=False):
        self.rows: dict[str, dict] = {}
        self.explode = explode

    def upsert(self, vectors, namespace):
        if self.explode:
            raise RuntimeError("pinecone is having a moment")
        for v in vectors:
            self.rows[v["id"]] = v["metadata"]

    def list(self, namespace):
        if self.explode:
            raise RuntimeError("pinecone is having a moment")
        yield FakePage(list(self.rows))

    def fetch(self, ids, namespace):
        return type("R", (), {
            "vectors": {i: FakeVector(self.rows[i]) for i in ids
                        if i in self.rows}})()


def log_with(index) -> QuestionLog:
    log = QuestionLog()
    log._index = index
    return log


def test_a_question_is_recorded_with_its_verdict():
    index = FakeIndex()
    log = log_with(index)
    asyncio.run(log.record("what did ansem say about eth", asker="lex",
                           answered=True, reference="111"))
    assert "q-111" in index.rows
    row = index.rows["q-111"]
    assert row["question"] == "what did ansem say about eth"
    assert row["answered"] is True
    assert row["asker"] == "lex"
    assert row["asked_at"]


def test_a_miss_is_recorded_as_one():
    """The misses are the reason this exists."""
    index = FakeIndex()
    asyncio.run(log_with(index).record("what did taylor swift say",
                                       answered=False, reference="222"))
    assert index.rows["q-222"]["answered"] is False


def test_the_same_mention_twice_is_one_row():
    """Retried after a restart, it must not count as two questions."""
    index = FakeIndex()
    log = log_with(index)
    asyncio.run(log.record("same question", reference="333"))
    asyncio.run(log.record("same question", reference="333"))
    assert len(index.rows) == 1


def test_an_empty_question_is_not_recorded():
    index = FakeIndex()
    asyncio.run(log_with(index).record("   ", reference="444"))
    assert index.rows == {}


def test_a_write_failure_never_reaches_the_caller():
    """The reply is already posted by the time this runs. A logging
    failure must not become the caller's problem."""
    log = log_with(FakeIndex(explode=True))
    asyncio.run(log.record("anything", reference="555"))   # must not raise


def test_a_read_failure_returns_nothing_rather_than_raising():
    assert asyncio.run(log_with(FakeIndex(explode=True)).list_all()) == []


def test_what_was_written_can_be_read_back():
    """The bug this file was written for: ids live on page.vectors, and
    reading the page directly returned empty forever without erroring."""
    index = FakeIndex()
    log = log_with(index)
    asyncio.run(log.record("first", reference="a"))
    asyncio.run(log.record("second", reference="b"))
    rows = asyncio.run(log.list_all())
    assert len(rows) == 2, "wrote two, read back none — the silent-hole bug"
    assert {r["question"] for r in rows} == {"first", "second"}


def test_newest_first():
    index = FakeIndex()
    index.rows = {
        "q-old": {"question": "old", "asked_at": "2026-01-01T00:00:00+00:00"},
        "q-new": {"question": "new", "asked_at": "2026-08-01T00:00:00+00:00"},
    }
    rows = asyncio.run(log_with(index).list_all())
    assert [r["question"] for r in rows] == ["new", "old"]


@pytest.mark.parametrize("source", ["x", "web"])
def test_the_surface_is_recorded(source):
    """Worth separating: a question typed on the site and one tagged in
    public are different signals about what people want."""
    index = FakeIndex()
    asyncio.run(log_with(index).record("q", source=source, reference="s"))
    assert index.rows["q-s"]["source"] == source
