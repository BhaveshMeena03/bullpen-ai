"""The exact-token index, and the promise that it can only add.

Semantic search is weak on rare tokens: a name or a number is one word in
four hundred and barely moves an embedding, so the passage that literally
contains it can rank below passages merely about the same subject. Three
misses in one day came from that.

The rule this file exists to hold is that the fix is strictly additive. It
contributes candidates to the pool the reranker already scores. It must
never reorder, never drop, and never be able to turn a working answer into
a worse one — retrieval quality is the product.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.terms import TermIndex  # noqa: E402

INDEX = ROOT / "data" / "term_index.json"


def test_the_shipped_index_covers_the_queries_that_failed():
    """Each of these missed in production, and each is in the archive."""
    terms = TermIndex()
    assert len(terms) > 1000, "the index looks empty"

    for query in ("what did andre say",
                  "what did mayne n ansem talk about",
                  "who made 54 million on the drop"):
        assert terms.lookup(query), query


def test_it_stays_out_of_the_way_of_questions_that_work():
    """A query with no rare token contributes nothing, so those searches
    are byte-for-byte what they were before this existed."""
    terms = TermIndex()
    for query in ("what did they say about pump fun fees",
                  "what did ansem say about solana"):
        assert terms.lookup(query) == [], query


def test_a_missing_index_is_not_an_error(tmp_path):
    """Absent or unreadable degrades to the old behaviour, silently and
    on purpose: search must not fail because an optimisation is missing."""
    assert TermIndex(tmp_path / "nope.json").lookup("what did andre say") == []

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert TermIndex(broken).lookup("what did andre say") == []


def test_a_bare_number_is_not_indexed_but_a_phrase_is():
    """"54" is in dozens of windows and says nothing; "54 million" is in
    a handful and is the whole question.

    The count is a band rather than an exact number. This asserted
    exactly one window and went red the day four more broadcasts were
    indexed and another episode happened to say "54 million" -- a true
    fact about a growing archive failing a test about the indexer. What
    the indexer promises is that a bare number is dropped and a phrase
    is kept and stays selective; how many windows say a given phrase is
    the corpus's business, not this file's.
    """
    raw = json.loads(INDEX.read_text())
    assert "54" not in raw["terms"], "a bare number carries no question"
    assert "54 million" in raw["terms"], "the phrase is the whole question"
    windows = raw["terms"]["54 million"]
    assert 1 <= len(windows) <= 20, (
        f"{len(windows)} windows: selective enough to be worth a lookup, "
        "and if this ever grows past the cap the phrase has stopped "
        "identifying a moment")


def test_lookups_are_capped():
    """This only needs to put a few candidates in front of the reranker.
    Flooding the pool would let a keyword match crowd out the embedding's
    own choices, which is the one thing it must not do."""
    terms = TermIndex()
    assert len(terms.lookup("kimchi", limit=3)) <= 3
    assert len(terms.lookup("kimchi")) <= 8


def test_rarest_terms_come_first():
    """A token in two windows says far more about what was meant than one
    in fifty, and only the first few survive the cap."""
    terms = TermIndex()
    raw = json.loads(INDEX.read_text())
    # "andre" is rare, "trading" is not; the rare one must lead.
    picked = terms.lookup("what did andre say about trading", limit=3)
    andre_ids = {raw["ids"][i] for i in raw["terms"]["andre"]}
    assert picked and picked[0] in andre_ids
