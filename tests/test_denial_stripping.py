"""Do not open by saying you found nothing, then find it.

    "I couldn't find a detailed introduction to Michael Cat in these
     episodes, but from context he appears to be the head of production
     at Market Bubble.

     Around 5:01 ... one of the hosts describes him as 'our head of
     production' and says 'He does 80 different jobs.'"

The answer is right. The first sentence says it is not, and on X that is
the only sentence most people read.

Rule 1a was written at this twice -- once as a principle, then again as a
mechanic listing the forbidden openings verbatim. It went from four in
fifteen to two and stopped there. Attribution went the same way until it
moved out of the prompt and into code, so this does too.

A genuine refusal is untouched, and the test is not what the sentence
says: it is whether the reply goes on to cite a moment. With nothing
cited, "I couldn't find that" is the honest answer and the whole reply.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.x_bot import strip_leading_denial  # noqa: E402


class TestRepliesThatActuallyWentOut:
    """Every one of these was observed, not invented."""

    def test_the_michael_catt_answer(self):
        before = ('I couldn\'t find a detailed introduction to Michael Cat '
                  'in these episodes, but from context he appears to be the '
                  'head of production at Market Bubble.\n\nAround 5:01 in '
                  '"Market Bubble: The Ansem Edition", one of the hosts '
                  'describes him as "our head of production" and says "He '
                  'does 80 different jobs."')
        after, changed = strip_leading_denial(before)
        assert changed
        assert after.startswith("Around 5:01")
        assert "couldn't find" not in after
        assert "80 different jobs" in after

    def test_a_however_turn_is_swallowed_with_it(self):
        """Removing the denial and leaving "However," reads worse than
        leaving both."""
        before = ("I don't see a direct statement from Ansem about that in "
                  "these transcripts. However, around 1:46:25 in Market "
                  "Bubble #4, Ansem discusses how he finds opportunities.")
        after, changed = strip_leading_denial(before)
        assert changed and after.startswith("Around 1:46:25")

    def test_a_denial_that_is_only_a_clause(self):
        """The answer arrives in the same sentence, after a turn."""
        before = ("Not in those words, but the archive has this: around "
                  "27:09 David Hoffman sold his entire Ethereum position "
                  "and said he is never buying again.")
        after, changed = strip_leading_denial(before)
        assert changed and "Not in those words" not in after
        assert "27:09" in after

    def test_a_hedged_summary(self):
        before = ("I don't have enough material for a full summary of "
                  "episode 17, but I can tell you what stands out. It aired "
                  "August 29 and around 0:31 Banks explains the draft.")
        after, changed = strip_leading_denial(before)
        assert changed and "don't have enough" not in after


class TestARealRefusalSurvives:
    @pytest.mark.parametrize("refusal", [
        "I couldn't find that in the episodes I've indexed.",
        "I looked, and couldn't find that in the episodes I've indexed.",
        "I couldn't find that in the episodes I've indexed. Try naming the "
        "guest or the episode?",
        "There's no mention of that in the transcripts I have.",
    ])
    def test_it_is_untouched(self, refusal):
        """Nothing is cited, so the denial IS the answer. Stripping here
        would leave a reply that says nothing at all."""
        after, changed = strip_leading_denial(refusal)
        assert not changed and after == refusal


class TestACleanAnswerIsNotTouched:
    @pytest.mark.parametrize("answer", [
        "Around 27:09 in Market Bubble #4, Ansem named David Hoffman.",
        "Ansem said Bonk went from $30 million to $4.5 billion, around 9:47.",
        "",
        "   ",
    ])
    def test_it_passes_through(self, answer):
        after, changed = strip_leading_denial(answer)
        assert not changed and after == answer

    def test_a_denial_with_no_citation_after_it_is_kept(self):
        """The guard is the citation, not the wording -- an answer that
        denies and then waffles has not actually found anything."""
        before = ("I couldn't find that in the episodes I've indexed. The "
                  "transcripts cover a lot of ground but nothing matching "
                  "what you asked about specifically here today.")
        after, changed = strip_leading_denial(before)
        assert not changed and after == before


class TestItRunsInTheReplyPath:
    source = (ROOT / "app" / "x_bot.py").read_text()

    def test_compose_calls_it(self):
        assert "strip_leading_denial(result.answer)" in self.source

    def test_it_runs_before_the_gates_that_judge_the_answer(self):
        """is_a_miss and salvage read the whole string, so stripping has
        to happen first or they judge text that will not be posted."""
        strip = self.source.index("strip_leading_denial(result.answer)")
        assert strip < self.source.index("rescued = salvage(result.answer)")

    def test_it_is_logged(self):
        """Silently rewriting a public reply is not acceptable."""
        assert "dropped a denial in front of a cited answer" in self.source

    def test_preview_reply_does_not_exercise_this_path(self):
        """A note, not a requirement. preview_reply.py calls
        index.search() directly and never runs compose(), so it shows the
        raw answer -- which is why a hedge still appears there and made
        this look unfixed once already.
        """
        preview = (ROOT / "scripts" / "preview_reply.py").read_text()
        assert "compose(" not in preview
