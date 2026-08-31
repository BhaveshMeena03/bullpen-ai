"""An answer must not open by denying what it goes on to say.

    "I couldn't find that in the episodes I've indexed. The excerpts
     mention Michael Cat repeatedly — described as the head of
     production at Market Bubble around 3:05."

Both halves went out. A reader on X takes the first line and scrolls, so
a reply that worked reads as a tool that did not -- and that is the reply
somebody screenshots when they are deciding whether this is worth
anything.

Rule 1a was aimed at this twice: once as a principle, then rewritten as a
mechanic listing the forbidden openings verbatim. It moved 4-in-15 to
2-in-15 and stopped there. Everything else that resisted the prompt this
way -- the joke pool, model-written links, quotes credited to the wrong
host -- ended up in code.

The two directions matter equally. A real refusal cites nothing and must
survive whole, because "I couldn't find that in the episodes I've
indexed" is the honest answer to a question the archive cannot answer,
and it is most of what keeps this account trustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.hedging import strip_denial  # noqa: E402


class TestDenialsThatWerePosted:
    """Every one of these is a live reply, not an invention."""

    @pytest.mark.parametrize("answer,gone", [
        ("I couldn't find that in the episodes I've indexed. The excerpts "
         "mention \"Michael Cat\" repeatedly—described as the head of "
         "production at Market Bubble around 3:05.",
         "I couldn't find"),
        ("I don't see a direct statement from Ansem about paying attention "
         "to crypto before everyone else in these transcripts. However, "
         "around 1:46:25 Ansem does discuss how he identifies trading "
         "opportunities early.",
         "I don't see"),
        ("I couldn't find a specific excerpt where Ansem discusses making a "
         "plan before you buy. Around 3:04:39 he laid out the levels he "
         "would bid into.",
         "I couldn't find"),
        ("Not in those words, but the archive has this: around 27:09 David "
         "Hoffman sold his entire Ethereum position.",
         "Not in those words"),
        ("I don't have enough material to give you a full summary of episode "
         "17, but I can tell you what stands out. The episode aired August "
         "29 and around 0:31 Banks explains the draft format.",
         "I don't have"),
        ("There is no specific mention of that. Around 9:37 Ansem called "
         "Hyperliquid a real business making money.",
         "no specific mention"),
    ])
    def test_the_denial_is_removed(self, answer, gone):
        out, removed = strip_denial(answer)
        assert removed and gone.lower() in removed.lower()
        assert gone.lower() not in out.lower()

    def test_the_citation_always_survives(self):
        """The timestamp is the whole product. Removing the denial must
        never take the evidence with it."""
        out, _ = strip_denial(
            "I couldn't find that in the episodes I've indexed. The excerpts "
            "mention Michael Cat as head of production around 3:05.")
        assert "3:05" in out

    def test_the_remainder_reads_as_a_sentence(self):
        """A stripped answer starting "however, around 1:46:25" reads like
        half of something."""
        out, _ = strip_denial(
            "I don't see a direct statement about that. However, around "
            "1:46:25 Ansem discusses how he finds opportunities early.")
        assert not out.lower().startswith("however")
        assert out[0].isupper()


class TestWhatMustSurviveUntouched:
    @pytest.mark.parametrize("answer", [
        # A real refusal. Cites nothing, so there is nothing to contradict.
        "I couldn't find that in the episodes I've indexed.",
        "I looked, and couldn't find that in the episodes I've indexed.",
        "I couldn't find that. Try naming the guest or the episode?",
        # A clean answer with no denial in it at all.
        "Around 27:09 in Market Bubble #4, Ansem named David Hoffman as the "
        "one who sold his entire position.",
        # A qualifier AFTER the answer, which rule 1a explicitly asks for.
        "Around 27:09 Ansem named David Hoffman, though he doesn't put it in "
        "those words.",
        # Too little left to be an answer.
        "I couldn't find that. Around 27:09.",
    ])
    def test_it_is_returned_unchanged(self, answer):
        out, removed = strip_denial(answer)
        assert removed is None
        assert out == answer

    def test_a_refusal_keeps_its_whole_meaning(self):
        """This is most of what makes the account trustworthy. Mangling it
        would be far worse than the bug being fixed."""
        refusal = "I couldn't find that in the episodes I've indexed."
        assert strip_denial(refusal) == (refusal, None)


class TestDegradesQuietly:
    @pytest.mark.parametrize("junk", ["", "   ", None])
    def test_nothing_in_nothing_out(self, junk):
        out, removed = strip_denial(junk)
        assert removed is None and out == junk

    def test_a_denial_with_no_citation_anywhere_is_kept(self):
        """Without a citation there is no evidence the denial is false, so
        there is no basis to remove it."""
        answer = ("I couldn't find that. The transcripts cover several "
                  "projects but nothing matching what you asked.")
        assert strip_denial(answer)[1] is None


class TestItRunsInTheReplyPath:
    source = (ROOT / "app" / "x_bot.py").read_text()

    def test_the_bot_calls_it(self):
        assert "hedging.strip_denial(result.answer)" in self.source

    def test_it_runs_before_the_gates_that_judge_the_answer(self):
        """is_a_miss and is_a_deflection read the whole string, so the
        denial has to be gone before they look at it -- otherwise an
        answer is judged on a sentence that is about to be deleted."""
        fix = self.source.index("hedging.strip_denial")
        assert fix < self.source.index("rescued = salvage(result.answer)")

    def test_a_removal_is_logged(self):
        """Silently editing a public reply is not acceptable; this has to
        be greppable afterwards."""
        assert "dropped a denial the answer contradicts" in self.source
