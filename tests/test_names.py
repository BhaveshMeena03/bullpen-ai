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


class TestItRunsInBothPaths:
    def test_retrieval_expands_the_term_lookup(self):
        source = (ROOT / "app" / "podcast.py").read_text()
        assert "names.expand(query)" in source

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
        is not worth having."""
        source = (ROOT / "app" / "podcast.py").read_text()
        assert "ids = ids or set()" in source
