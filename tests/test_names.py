"""Spell the names right, even when the captions do not.

Whisper mishears the words this archive is most about. Counted across
every transcript: "Salana" for Solana 430 times of 1,131, "poly market"
151 of 268, "Hyperlid" plus "hyper liquid" 229 of 526, "pumpfun" 43 of
44, "Frentech" and "Mount Gox" every single time.

Two consequences, and the second is the one people see.

The exact-token index matches letters, so it has holes exactly where the
captions do -- a lookup for "solana" cannot see four hundred lines that
say "Salana". The embeddings bridge that on their own, which is why a
search for friend.tech found "Frentech"; the term index cannot.

And a quoted mangling goes out under a real person's name. "Hyperlid
briefly flipped Salana price" was posted as FaZe Banks' words. He said
Hyperliquid and Solana, so reproducing the transcription error IS the
misquote, and correcting it is the faithful thing rather than a liberty.

What is deliberately NOT corrected matters as much. "per" appears 170
times and is almost always the English word rather than PURR; "Seoul" 13
times and is a city as often as it is SOL; "JTO" is Jito's actual
ticker. A wrong correction is worse than an uncorrected error, because
it is one this code chose.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.names import expand, fix  # noqa: E402


class TestTheManglingsThatWerePosted:
    @pytest.mark.parametrize("mangled,correct", [
        ('FaZe Banks said "Hyperlid briefly flipped Salana price"',
         "Hyperliquid briefly flipped Solana price"),
        ("Ansem said pumpfun made a billion dollars in fees",
         "pump.fun made a billion"),
        ("He mentioned Frentech around 2:08", "friend.tech around 2:08"),
        ("the Mount Gox story", "Mt. Gox story"),
        ("They discussed poly market odds", "Polymarket odds"),
        ("z cash is at all-time highs", "Zcash is at all-time highs"),
        ("Corweave is up 10x", "CoreWeave is up 10x"),
        ("Lucanets coming on the show", "Luca Netz coming on the show"),
    ])
    def test_it_is_corrected(self, mangled, correct):
        assert correct in fix(mangled)[0]

    def test_what_changed_is_reported(self):
        """Editing a public reply silently is not acceptable; the caller
        logs this."""
        _, changed = fix("Hyperlid flipped Salana")
        assert len(changed) == 2

    def test_shouting_survives(self):
        """A transcript in caps should not force canonical case into
        prose, nor be silently lowercased."""
        assert "SOLANA" in fix("SALANA IS RIPPING")[0]


class TestWhatMustNotBeTouched:
    @pytest.mark.parametrize("safe", [
        # "per" is 170 hits and nearly always the English word.
        "he charges 2% per month on the loans",
        "a hundred dollars per person",
        # Seoul is a real city, 13 hits, as often as it is SOL.
        "were you in Seoul with TJR",
        # JTO is Jito's actual ticker.
        "the JTO airdrop was big",
        # Already correct.
        "Solana is good with marketing",
        "Around 27:09 Ansem named David Hoffman.",
        "Hyperliquid is at all-time highs",
    ])
    def test_it_is_returned_unchanged(self, safe):
        out, changed = fix(safe)
        assert out == safe and changed == []

    @pytest.mark.parametrize("junk", ["", None])
    def test_nothing_in_nothing_out(self, junk):
        assert fix(junk) == (junk, [])


class TestQueryExpansion:
    def test_a_search_reaches_the_mangled_spelling(self):
        """Somebody searching "solana" should reach the four hundred
        lines that say "Salana"."""
        out = expand("what did ansem say about solana")
        assert any("salana" in q for q in out)

    def test_hyperliquid_reaches_both_of_its_manglings(self):
        out = expand("hyperliquid all time highs")
        assert any("hyperlid" in q for q in out)
        assert any("hyper liquid" in q for q in out)

    def test_a_query_with_no_alias_expands_to_nothing(self):
        """No extra lookups for a question that has no mangled form."""
        assert expand("who was the guy who sold his eth") == []

    def test_it_does_not_expand_what_is_already_mangled(self):
        assert expand("what did ansem say about salana") == []


class TestRetrievalActuallyRunsTheExpansion:
    """Calling it, not grepping for it.

    The first version of this class asserted the string
    "names.expand(query)" appeared in podcast.py. It did appear, and the
    line beside it raised TypeError on every query containing a name --
    lookup() returns a list and the code unioned it like a set. The test
    passed; search 500'd in production for Solana, Hyperliquid, pump.fun
    and Luca Netz until somebody asked one out loud.

    A test that reads source can only prove the code is present. These
    run it.
    """

    def _index(self, ids_by_spelling):
        """A PodcastIndex whose term lookup returns what this test says,
        without touching Pinecone or the network."""
        from app.podcast import PodcastIndex

        index = object.__new__(PodcastIndex)
        index._terms = type("Terms", (), {
            "lookup": staticmethod(
                lambda q, limit=8: list(ids_by_spelling.get(q.lower(), []))),
        })()
        return index

    def test_a_name_query_does_not_raise(self):
        """The production 500. "Luca Netz" expands to "lucanets", and the
        second lookup was unioned into the first with |=."""
        index = self._index({"what did luca netz talk about": ["a", "b"]})
        assert asyncio.run(index._add_exact_matches(
            "what did luca netz talk about", [])) == []

    def test_the_mangled_spelling_is_looked_up_too(self):
        """The point of the expansion: a query saying "solana" has to reach
        the four hundred passages that say "Salana"."""
        asked = []
        index = self._index({})
        index._terms.lookup = lambda q, limit=8: asked.append(q.lower()) or []
        asyncio.run(index._add_exact_matches("what did ansem say about solana", []))
        assert any("salana" in q for q in asked)

    def test_the_rarest_first_order_survives(self):
        """lookup() returns a list ranked by rarity. A set would discard
        that ranking, and the ranking is the whole value of the index."""
        from app import podcast

        seen = {}
        index = self._index({})
        index._terms.lookup = lambda q, limit=8: (
            ["rare", "common"] if "salana" not in q else ["mangled"])
        index._index = type("I", (), {
            "fetch": staticmethod(
                lambda ids, namespace: seen.update(order=list(ids)) or
                type("F", (), {"vectors": {}})())})()
        index._settings = type("S", (), {"pinecone_read_timeout_seconds": 5})()
        asyncio.run(index._add_exact_matches("what did ansem say about solana", []))
        assert seen["order"] == ["rare", "common", "mangled"]

    def test_expansion_cannot_flood_the_reranker(self):
        """Before expansion existed one lookup contributed at most eight
        candidates. Expanding a query must not quietly raise that."""
        from app.podcast import _EXACT_MATCH_CAP

        index = self._index({})
        index._terms.lookup = lambda q, limit=8: [
            f"{q[:4]}-{n}" for n in range(8)]
        index._index = type("I", (), {"fetch": staticmethod(
            lambda ids, namespace: (_ for _ in ()).throw(
                AssertionError(f"{len(ids)} ids, cap is {_EXACT_MATCH_CAP}"))
            if len(ids) > _EXACT_MATCH_CAP else type("F", (), {"vectors": {}})()
        )})()
        index._settings = type("S", (), {"pinecone_read_timeout_seconds": 5})()
        asyncio.run(index._add_exact_matches("solana and hyperliquid and pump.fun", []))

    def test_a_broken_lookup_degrades_to_the_vector_results(self):
        """The docstring promises this. Until now only the fetch was
        wrapped, so a failure in the lookup above it became a 500 instead
        of the search it would have been without the term index at all."""
        index = self._index({})
        index._terms.lookup = lambda q, limit=8: 1 / 0
        existing = ["the vector results"]
        assert asyncio.run(index._add_exact_matches("solana", existing)) is existing


class TestItRunsInBothPaths:
    def test_the_reply_corrects_the_spelling(self):
        source = (ROOT / "app" / "x_bot.py").read_text()
        assert "names.fix(result.answer)" in source

    def test_the_correction_runs_before_the_answer_is_judged(self):
        """The gates below read the whole string, so the spelling has to
        be settled before they look at it."""
        source = (ROOT / "app" / "x_bot.py").read_text()
        assert source.index("names.fix") < source.index(
            "rescued = salvage(result.answer)")

    def test_a_failed_term_lookup_still_degrades_to_the_old_behaviour(self):
        """Retrieval quality is the product; an addition that can subtract
        is not worth having. Proven by running it, in
        TestRetrievalActuallyRunsTheExpansion above -- this used to assert
        the source contained "ids = ids or set()", which was the line that
        crashed."""
        source = (ROOT / "app" / "podcast.py").read_text()
        assert "exact-match lookup failed" in source
