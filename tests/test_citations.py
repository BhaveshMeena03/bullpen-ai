"""Move a citation only when the passages prove it is wrong.

The corrector exists because SYSTEM_PROMPT rule 2 already forbids citing
the `at` attribute of an excerpt instead of the line actually used, and
the model does it anyway. The risk is the mirror image: a corrector that
fires on a citation that was right is worse than the drift, because the
drift is occasional and a bad corrector is systematic.

Every "leave it alone" case below is a shape measured on the live index,
where an earlier version of this module moved a correct citation:

    a host repeats a phrase, so the quote matches at 18:10 AND 21:23;
    the answer cited 21:30 and was right

    an answer writes "1:49" for one hour forty-nine on a three-hour
    show, and reading it as 1m49s finds nothing near second 109

    two quotes in one sentence leave prose between them, and the quote
    regex pairs the first closing mark with the second opening one
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import citations  # noqa: E402


class _Hit:
    def __init__(self, text_ts: str) -> None:
        self.text_ts = text_ts


def test_moves_a_citation_the_quote_contradicts():
    """The measured failure: cited 1:15:21, said at 1:18:49."""
    hits = [_Hit(
        "[1:15:20] Ansem: pump fun is sitting right under resistance\n"
        "[1:18:49] Ansem: i think the meme coins pair with stocks now\n"
    )]
    answer = ('Around 1:15:21 he says "i think the meme coins pair '
              'with stocks now".')
    fixed, changes = citations.correct(answer, hits)
    assert "1:18:49" in fixed
    assert "1:15:21" not in fixed
    assert len(changes) == 1


def test_leaves_a_correct_citation_alone():
    hits = [_Hit("[0:42:10] Ansem: leverage is how good traders go broke\n")]
    answer = 'At 42:02 he says "leverage is how good traders go broke".'
    fixed, changes = citations.correct(answer, hits)
    assert fixed == answer and changes == []


def test_keeps_a_citation_when_the_host_repeats_himself():
    """Both lines carry the quote; the answer cited the later one.

    The earlier copy scores higher here, exactly as it did live, where
    the transcript redacts the expletive in one place and not the other.
    Best-match would move this. Support-at-the-claim keeps it.
    """
    hits = [_Hit(
        "[18:10] like alt seasons where a ton of alts did really well "
        "and went up a fuck ton.\n"
        "[21:23] ton of alts did really well and went up a [expletive] "
        "ton. But during those alt seasons there were also altcoins\n"
    )]
    answer = ('Around 21:30 he says "a ton of alts did really well and '
              'went up a fuck ton".')
    fixed, changes = citations.correct(answer, hits)
    assert fixed == answer and changes == []


def test_reads_a_short_stamp_as_hours_when_that_is_what_supports_it():
    """"1:49" on a three-hour show is 1h49m, not 1m49s."""
    hits = [_Hit(
        "[1:49:37] ton of like super legit big boy market makers on "
        "solana and so yeah it's just crazy to see\n"
    )]
    answer = ('Around 1:49 he mentions "a ton of like super legit big '
              'boy market makers on solana".')
    fixed, changes = citations.correct(answer, hits)
    assert fixed == answer and changes == []


def test_ignores_prose_the_quote_regex_paired_by_accident():
    """Text between two quotations is not a quotation.

    It carries a timestamp, which no line of transcript does, and that
    is what marks it as the answer's own narration.
    """
    said = " in the next year, and around 29:04 he compared the two"
    assert citations.moment_of(said, [(1000, said)]) is None


def test_ignores_a_paraphrase():
    hits = [_Hit("[0:30:00] Ansem: i think we chop sideways for a while\n")]
    answer = "At 30:00 he expects sideways price action."
    fixed, changes = citations.correct(answer, hits)
    assert fixed == answer and changes == []


def test_ignores_a_quote_no_line_carries():
    hits = [_Hit("[0:30:00] Ansem: i think we chop sideways for a while\n")]
    answer = 'At 30:00 he says "bitcoin reaches four hundred thousand".'
    fixed, changes = citations.correct(answer, hits)
    assert fixed == answer and changes == []


def test_offers_the_hour_reading_only_where_it_is_plausible():
    """"1:49" is ambiguous. "42:02" is not."""
    assert citations.readings("1:49") == [109, 6540]
    assert citations.readings("42:02") == [2522]
    assert citations.readings("1:18:49") == [4729]


def test_writes_the_second_the_way_a_person_does():
    assert citations._as_written(304) == "5:04"
    assert citations._as_written(4729) == "1:18:49"


def test_an_answer_with_no_passages_is_returned_untouched():
    answer = 'At 30:00 he says "something or other happened".'
    assert citations.correct(answer, [])[0] == answer
    assert citations.correct(answer, None)[0] == answer
