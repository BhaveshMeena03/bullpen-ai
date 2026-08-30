"""Never name the wrong host.

A reply carrying both hosts' words is normal and stays that way -- it is
a conversation and they talk over each other for four hours. What is not
normal is one man's sentence in the other's mouth:

    reply:       "Ansem said 'I own none of the token. I have been
                  sidelined.'"
    transcript:  FaZe Banks said it, 0:40:17

That is the $ANSEM token, so the false version has Ansem disclaiming
ownership of his own coin. Measured across a hundred questions, two
replies in a hundred did this -- always crediting the host the QUESTION
named rather than the one the transcript labels, which is the exact case
rule 5b spells out. Seven prompt rules did not close it.

The guard demotes rather than corrects. Choosing the other host would be
a second guess and a confident wrong correction is worse than a vague
true one, so "Ansem said" becomes "one of the hosts said" and the quote,
timestamp, episode and link are left exactly as they were.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.attribution import correct, labelled_lines, speaker_of  # noqa: E402


class Hit:
    """Only text_ts matters here -- it is the labelled copy the model reads."""

    def __init__(self, text_ts: str) -> None:
        self.text_ts = text_ts


REAL = Hit(
    "[0:40:17] FaZe Banks: clearly right here, right now. I own none of "
    "the token. I have been sidelined.\n"
    "[1:37:53] Ansem: I think Solana is good with marketing to be honest.\n"
    "[1:38:02] the guest answered without a name on the line at all"
)


class TestTheTwoRealFailures:
    """Both taken verbatim from a hundred-question run and verified
    against the voice-labelled transcript before being written down."""

    def test_banks_line_credited_to_ansem_is_demoted(self):
        out, changed = correct(
            'Around 40:17, Ansem said "I own none of the token. I have '
            'been sidelined."', [REAL])
        assert "one of the hosts said" in out
        assert "Ansem said" not in out
        assert changed and "FaZe Banks" in changed[0]

    def test_ansem_line_credited_to_banks_is_demoted(self):
        out, changed = correct(
            'Around 1:37:53, Banks said "I think Solana is good with '
            'marketing to be honest."', [REAL])
        assert "one of the hosts said" in out
        assert len(changed) == 1

    def test_the_quote_and_the_timestamp_survive(self):
        """Only the name was wrong. Removing the evidence would make the
        reply worse, not safer."""
        out, _ = correct(
            'Around 40:17, Ansem said "I own none of the token. I have '
            'been sidelined."', [REAL])
        assert '"I own none of the token. I have been sidelined."' in out
        assert "40:17" in out


class TestItLeavesCorrectRepliesAlone:
    @pytest.mark.parametrize("answer", [
        'Around 40:17, FaZe Banks said "I own none of the token. I have '
        'been sidelined."',
        'Around 1:37:53, Ansem said "I think Solana is good with marketing '
        'to be honest."',
        'Around 40:17, Banks said "I own none of the token. I have been '
        'sidelined."',                       # "Banks" == "FaZe Banks"
    ])
    def test_a_true_credit_is_untouched(self, answer):
        out, changed = correct(answer, [REAL])
        assert out == answer and changed == []

    def test_an_unquoted_paraphrase_is_untouched(self):
        """No quotation marks, nothing to check against a line. Acting
        here would demote answers that were never claims about wording."""
        answer = "Around 40:17 Ansem explained his position on the token."
        assert correct(answer, [REAL]) == (answer, [])

    def test_a_guest_is_never_demoted(self):
        """Only the two hosts are voice-labelled; every guest line is
        unprefixed. Touching those would take 'what did Jesse say about
        Base' from a good answer to a vague one."""
        answer = 'Jesse said "Base is the consumer chain" around 1:06.'
        assert correct(answer, [REAL]) == (answer, [])

    def test_both_hosts_in_one_reply_is_fine(self):
        """The thing that must NOT be broken: it is their conversation."""
        answer = ('Ansem said "I think Solana is good with marketing to be '
                  'honest." FaZe Banks said "I own none of the token. I '
                  'have been sidelined."')
        assert correct(answer, [REAL]) == (answer, [])

    def test_two_wrong_credits_are_both_fixed(self):
        answer = ('Banks said "I think Solana is good with marketing to be '
                  'honest." Ansem said "I own none of the token. I have '
                  'been sidelined."')
        out, changed = correct(answer, [REAL])
        assert len(changed) == 2
        assert "Banks said" not in out and "Ansem said" not in out


class TestDegradesQuietly:
    @pytest.mark.parametrize("hits", [None, [], [Hit("")], [Hit("no labels here")]])
    def test_nothing_to_check_against_changes_nothing(self, hits):
        answer = 'Ansem said "something entirely unmatched by any line."'
        assert correct(answer, hits) == (answer, [])

    def test_an_empty_answer_is_survived(self):
        assert correct("", [REAL]) == ("", [])

    def test_a_quote_matching_no_line_is_left_alone(self):
        """None means "not established", never "wrong" -- this only acts
        on a disagreement it can demonstrate."""
        answer = 'Ansem said "a sentence that appears in no transcript at all."'
        assert correct(answer, [REAL]) == (answer, [])

    def test_a_short_quote_is_not_matched_on_noise(self):
        answer = 'Ansem said "the token."'
        assert correct(answer, [REAL]) == (answer, [])


class TestParsing:
    def test_it_reads_both_hosts_off_the_labelled_lines(self):
        assert set(dict(labelled_lines([REAL]))) == {"FaZe Banks", "Ansem"}

    def test_an_unlabelled_line_is_not_a_speaker(self):
        assert all(who in ("FaZe Banks", "Ansem")
                   for who, _ in labelled_lines([REAL]))

    def test_speaker_of_finds_the_right_line(self):
        lines = labelled_lines([REAL])
        assert speaker_of("I own none of the token. I have been sidelined",
                          lines) == "FaZe Banks"
        assert speaker_of("Solana is good with marketing to be honest",
                          lines) == "Ansem"

    def test_speaker_of_returns_none_when_unsure(self):
        assert speaker_of("nothing like any line here", labelled_lines([REAL])) is None


class TestItRunsInTheReplyPath:
    source = (ROOT / "app" / "x_bot.py").read_text()

    def test_the_bot_calls_it(self):
        assert "attribution.correct(result.answer, result.hits)" in self.source

    def test_it_runs_before_the_gates_that_judge_the_answer(self):
        """What is measured, logged and posted has to be one string."""
        fix = self.source.index("attribution.correct")
        assert fix < self.source.index("rescued = salvage(result.answer)")

    def test_a_correction_is_logged_loudly(self):
        """Silent rewriting of a public reply is not acceptable; this has
        to be greppable after the fact."""
        assert "attribution corrected" in self.source
