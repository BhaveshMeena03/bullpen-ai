"""Never name a recording the search did not return.

The Musk archive is the one the model has heard before, so it can name
a plausible interview from memory and does. Every case below is a shape
measured over twenty-five questions, where six of the eight recall
answers cited a recording that was never retrieved.

The untouched cases matter as much as the demoted ones. This runs on
every reply, and an archive of eleven recordings gives an answer plenty
of chances to name a real one correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import sources  # noqa: E402


class _Hit:
    def __init__(self, title: str, published_at: str) -> None:
        self.title = title
        self.published_at = published_at


LEX_2024 = _Hit("Elon Musk: Neuralink and the Future of Humanity | Lex "
                "Fridman Podcast #438", "2024-08-02")
LEX_2019 = _Hit("Elon Musk: Tesla Autopilot | Lex Fridman Podcast #18",
                "2019-04-12")
ROGAN_2021 = _Hit("Joe Rogan Experience #1609 - Elon Musk", "2021-02-11")


def test_demotes_a_recording_that_never_came_back():
    """Shown only 2024 Lex Fridman, it cited a 2021 Joe Rogan episode."""
    answer = ('Around 7:00 in the 2021 Joe Rogan conversation he says '
              'he sleeps about six hours.')
    fixed, changes = sources.correct(answer, [LEX_2024])
    assert "Joe Rogan" not in fixed
    assert "one of these conversations" in fixed
    assert len(changes) == 1


def test_demotes_a_year_and_show_it_was_given_but_never_together():
    """The pairing failure: both halves present, the recording is not.

    Shown 2021 Joe Rogan and 2019 Lex Fridman, an answer cited "the
    2021 Lex Fridman episode". Checking years and shows separately
    passes this, which is why the guard checks the pair.
    """
    answer = 'Around 31:19 in the 2021 Lex Fridman episode he explains why.'
    fixed, changes = sources.correct(answer, [ROGAN_2021, LEX_2019])
    assert "one of these conversations" in fixed
    assert len(changes) == 1


def test_leaves_a_recording_that_did_come_back():
    answer = 'Around 24:40 in the 2024 Lex Fridman episode he describes it.'
    fixed, changes = sources.correct(answer, [LEX_2024])
    assert fixed == answer and changes == []


def test_leaves_the_show_named_the_other_way_round():
    answer = 'In the Lex Fridman conversation from 2024 he describes it.'
    fixed, changes = sources.correct(answer, [LEX_2024])
    assert fixed == answer and changes == []


def test_demotes_a_recording_named_by_date_alone():
    answer = 'By 2020 he described it differently in the 2020 episode.'
    fixed, changes = sources.correct(answer, [LEX_2024])
    assert "2020 episode" not in fixed
    assert len(changes) == 1


def test_a_back_reference_keeps_reading_like_english():
    """"that same 2021 conversation" must not become "that same one of
    these conversations"."""
    answer = ('Around 7:00 in the 2021 Joe Rogan conversation he says six '
              'hours, and in that same 2021 conversation he mentions work.')
    fixed, _ = sources.correct(answer, [LEX_2024])
    assert "that same conversation" in fixed
    assert "same one of these" not in fixed


def test_an_answer_naming_no_recording_is_untouched():
    answer = 'He says he sleeps about six hours and works most weekends.'
    assert sources.correct(answer, [LEX_2024])[0] == answer


def test_no_passages_means_no_opinion():
    answer = 'Around 7:00 in the 2021 Joe Rogan conversation he says six.'
    assert sources.correct(answer, [])[0] == answer
    assert sources.correct(answer, None)[0] == answer
