"""The attribution checker has to be able to fail.

A verifier that reports "0 wrong" is only worth something if it would
have said otherwise on a wrong answer. Removing the prompt rule it
guards produced 0 failures too — the model simply did not misattribute
on that run — which leaves the checker itself unproven.

So these feed it the real failure directly: a passage where a line is
prefixed "Ansem:" and an answer that credits the line to Banks, exactly
as it happened when the speaker filter was first switched on.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_attribution import ATTRIBUTION, LABELLED, QUOTED, words  # noqa: E402

PASSAGE = (
    "[1:37:47] FaZe Banks: You get crushed, bro.\n"
    "[1:37:53] Ansem: I think Solana is good with marketing to be honest.\n"
    "[1:37:56] Ansem: they were in a really tough spot in 2022.\n"
    "[1:38:10] FaZe Banks: I was at the hackathons that whole year.\n"
)


def credited_speaker(answer: str, quote: str) -> str | None:
    """Who the answer credits for a quote — the checker's own logic."""
    where = answer.find(quote)
    credits = ATTRIBUTION.findall(answer[:where])
    if not credits:
        return None
    name = credits[-1][0]
    return "FaZe Banks" if name in ("Banks", "FaZe Banks") else name


def line_speaker(quote: str) -> str | None:
    target = words(quote)
    best, overlap = None, 0
    for speaker, text in LABELLED.findall(PASSAGE):
        shared = len(target & words(text))
        if shared > overlap:
            best, overlap = speaker, shared
    return best if overlap >= 3 else None


def test_the_real_misattribution_is_caught():
    """Verbatim shape of the reply that made me pull the filter."""
    answer = ('Banks said "I think Solana is good with marketing to be '
              'honest" during the July 9 episode.')
    quote = QUOTED.findall(answer)[0]
    assert line_speaker(quote) == "Ansem"
    assert credited_speaker(answer, quote) == "FaZe Banks"
    assert line_speaker(quote) != credited_speaker(answer, quote), (
        "the checker cannot tell these apart, so it cannot catch the bug")


def test_a_correct_attribution_passes():
    answer = ('Ansem said "I think Solana is good with marketing to be '
              'honest" during the July 9 episode.')
    quote = QUOTED.findall(answer)[0]
    assert line_speaker(quote) == credited_speaker(answer, quote) == "Ansem"


def test_banks_and_faze_banks_are_the_same_person():
    """The prefix says "FaZe Banks", answers say "Banks". Treating those
    as different people would make every correct answer look wrong."""
    answer = 'Banks said "I was at the hackathons that whole year" back then.'
    quote = QUOTED.findall(answer)[0]
    assert line_speaker(quote) == "FaZe Banks"
    assert credited_speaker(answer, quote) == "FaZe Banks"


def test_a_quote_from_an_unlabelled_line_is_not_judged():
    """Half the archive has no labels. Guessing there would invent
    failures, and inventing failures is how a check gets ignored."""
    assert line_speaker("something nobody in this passage ever said") is None


def test_an_unattributed_quote_is_not_judged():
    answer = 'One of the hosts said "I think Solana is good with marketing".'
    quote = QUOTED.findall(answer)[0]
    assert credited_speaker(answer, quote) is None


def test_the_attribution_pattern_reads_every_verb_people_use():
    for verb in ("said", "noted", "explained", "argued", "recalled",
                 "admitted", "claimed", "stated", "revealed"):
        assert ATTRIBUTION.search(f"Ansem {verb} that the market moved"), verb


def test_a_name_far_from_the_verb_is_not_an_attribution():
    """"Banks was present, and later somebody said" is not Banks saying
    it. The gap between name and verb is capped for that reason."""
    text = ("Banks was in the room for a long stretch of that conversation "
            "about several unrelated things, and then somebody said")
    assert not ATTRIBUTION.search(text)
