"""What this account will not post, whatever the model picked.

Every case below is real output. The prompt rule about jokes was loosened
to allow what the hosts say to each other — which is correct, that is the
show — and on the next run the model returned a suicide joke, a stabbing
bit, a slur, an antidepressant gag, and one host calling another a liar
by name. All five satisfied the loosened rule: a host said it, in the
room, with everyone trading it back.

That is the point of this filter. The prompt asks; this decides. These
entries post unprompted, under the show's name, with nobody reading them
first, so the last word cannot belong to the thing generating them.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402
from make_highlights import safe_to_post  # noqa: E402

# Verbatim from the run that prompted the filter.
CAME_BACK_FROM_THE_MODEL = [
    'Ansem jokingly threatens self-harm over market pressure, saying he\'ll '
    '"go fucking kill myself" by finding "the tallest building and jump off '
    'of it" to "shake the market".',
    'Banks riffs on a hypothetical London stabbing scenario, joking "I\'m '
    'six, five blue eyes, fucking hammer" before immediately admitting '
    '"I\'m just kidding."',
    'Banks immediately admits "I\'m just being retarded."',
    'Miz jokingly claims his net worth investment is "Lexa Pro" before '
    'laughing it off as a joke about himself',
    'Banks calls out Anom for lying about his old follower count, saying '
    '"You straight up lied."',
]


@pytest.mark.parametrize("said", CAME_BACK_FROM_THE_MODEL)
def test_what_the_loosened_rule_let_through_is_still_refused(said):
    assert not safe_to_post(said)


@pytest.mark.parametrize("said", [
    "Ansem says he'll kill himself if the trade goes against him",
    "Banks jokes about jumping off a bridge after the liquidation",
    "a guest says he wants to end it all over his portfolio",
])
def test_self_harm_is_refused_in_every_joking_register(said):
    """It is always framed as a joke. That is the framing this account
    would be repeating, on its own, as a joke."""
    assert not safe_to_post(said)


@pytest.mark.parametrize("said", [
    "Ansem describes getting robbed at gunpoint in Miami",
    "Mert recalls being mugged after a conference",
    "a guest talks about his brother's overdose",
    "Banks mentions his cancer scare last year",
    "Luca says he was in rehab that summer",
])
def test_crime_illness_and_addiction_stay_refused(said):
    """The old rule already said this. It said it in prose, to a model,
    which is not the same as it being true of the output."""
    assert not safe_to_post(said)


@pytest.mark.parametrize("said", [
    'Banks says Anom straight up lied about his follower count',
    'Ansem calls the project a scam and its founder a scammer',
    'Mayne says the team rugged everyone who bought in',
])
def test_a_named_person_accused_of_dishonesty_is_refused(said):
    """Fine as banter between people who know each other. This account
    republishing it alone, as a joke, is a different sentence."""
    assert not safe_to_post(said)


@pytest.mark.parametrize("said", [
    'Ansem declares he\'ll flip Mr. Beast "by the end of the year" because '
    'he has a "supercharged asset" Mr. Beast doesn\'t have.',
    'Easy\'s basketball argument ends with him admitting defeat: "I fought '
    'tooth and nail against this being a travel. It was a travel."',
    'Camila disclosed she made $54 million on "the drop."',
    'Luca revealed he sold Artifact to Nike for $2.5 million as a 10th '
    'grader.',
    'Greg said Agent usage has spiked so fast that "2.4 million people" '
    'signed up.',
])
def test_the_ones_worth_posting_still_pass(said):
    """A filter that refuses everything is not a filter."""
    assert safe_to_post(said)


def test_the_pool_that_is_live_right_now_passes():
    """Written before the filter existed. If one of these fails, it has
    been going out unprompted this whole time."""
    import json

    pool = json.loads((ROOT / "data" / "highlights.json").read_text())
    assert pool, "the pool is empty"
    bad = [h["text"] for h in pool if not safe_to_post(h.get("text", ""))]
    assert not bad, f"live entries this filter refuses: {bad}"


def test_empty_input_is_not_treated_as_safe_by_accident():
    assert safe_to_post("")  # nothing to refuse; the caller drops it earlier
