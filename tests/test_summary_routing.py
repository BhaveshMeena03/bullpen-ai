"""A summary request must reach the stored summary, not retrieval.

Every episode has a summary written once from the whole transcript and
kept. A request that misses the trigger falls through to ordinary search
and gets answered from six passages instead -- a partial account of a
three-hour show, assembled from whatever ranked, while the real thing
sits ready.

Nine phrasings in fourteen used to fall through, "tldr" among them,
which is the common way to ask on X.

The trigger widened, but not by much. It has to be a word that only ever
means "summarise": "about" would have caught "what did they say about
hyperliquid in episode 17" and returned a summary where somebody wanted
an answer, which is a worse failure than the one being fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.x_bot import asks_for_the_latest, summary_request  # noqa: E402


class TestNumberedRequestsReachTheStoredSummary:
    @pytest.mark.parametrize("asked", [
        "summarize episode 17", "summarise ep 17", "summary of episode 17",
        "can you summarize episode 17", "recap ep 17", "rundown of episode 17",
        "what happened in episode 17",
        # The ones that used to fall through.
        "tldr episode 17", "tl;dr of episode 17", "give me a tldr of ep 17",
        "brief me on episode 17", "overview of episode 17",
        "break down episode 17", "breakdown of ep 17", "sum up episode 17",
    ])
    def test_it_resolves_the_number(self, asked):
        assert summary_request(asked) == 17, asked

    def test_a_two_digit_number_is_read_whole(self):
        assert summary_request("tldr episode 16") == 16
        assert summary_request("summarize ep 9") == 9


class TestTheLatestEpisode:
    @pytest.mark.parametrize("asked", [
        "summarize the latest episode", "summarise the newest episode",
        "tldr of the latest episode", "recap the most recent episode",
        "what happened in the last show",
    ])
    def test_it_is_recognised_without_a_number(self, asked):
        """The model cannot know what is newest from six excerpts, so it
        must not be the one deciding."""
        assert asks_for_the_latest(asked), asked


class TestOrdinaryQuestionsAreNotHijacked:
    """The regression this widening could have caused. A summary returned
    where somebody asked a question is worse than the miss being fixed."""

    @pytest.mark.parametrize("asked", [
        "what did they say about hyperliquid in episode 17",
        "what did ansem say about solana in ep 16",
        "what did banks say about kick",
        "who was the guy who sold all his eth holding",
        "what did ansem say about the 17 million dollar wallet",
        "what did they say about the 2017 cycle",
    ])
    def test_it_still_goes_to_search(self, asked):
        assert summary_request(asked) is None, asked
        assert not asks_for_the_latest(asked), asked


class TestTheStoredSummaryIsTheLongestCut:
    def test_an_unnumbered_broadcast_can_win(self):
        """A show exists as a live broadcast and a shorter YouTube cut.
        The broadcast runs about twice as long and its summary covers the
        third of the show the upload never carries -- so "episode 17"
        must reach the broadcast, not the cut."""
        from app.x_bot import _same_show_as, episode_number
        rows = [
            {"title": "Polymarket Fantasy Football Draft | Market Bubble #17",
             "published_at": "2026-08-29", "summary": "x" * 1296},
            {"title": "$100K POLYMARKET FANTASY FOOTBALL DRAFT NIGHT",
             "published_at": "2026-08-27", "summary": "y" * 3114},
        ]
        matches = [r for r in rows if episode_number(r["title"]) == 17]
        matches += _same_show_as(matches, rows)
        best = max(matches, key=lambda s: len(s["summary"]))
        assert len(best["summary"]) == 3114
