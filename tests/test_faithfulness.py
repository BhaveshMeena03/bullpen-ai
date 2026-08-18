"""The hallucination checker, checked.

Every case below is a real one. This module shipped four wrong answers in a
row during its first afternoon, and each produced a confident accusation
against a product that had done nothing wrong:

  1. episode titles quoted correctly, reported as fabricated quotes
  2. the prose BETWEEN two quotations parsed as if it were a quotation
  3. an ellipsis or a [expletive] substitution scored as a mismatch
  4. a caption missing a word ("communism by another", "killing two birds
     one stone") scored at 84% and 56%, so the model was blamed for
     restoring what the caption had dropped

A checker that cries wolf is worse than no checker: it trains you to skim
past the report, and the one genuine finding then goes out with the noise.
So the tests that matter most here are the ones asserting silence.

The other half is the opposite risk. A checker tuned until nothing fails is
also worthless, so every loosening above is paired with a fabrication that
must still be caught — including one hidden behind an ellipsis, which is the
obvious way to smuggle an invented clause past a fragment-wise comparison.
"""

import pytest

from evals.faithfulness import (
    audit,
    check_quotes,
    coverage,
    extract_figures,
    extract_quotes,
    extract_timestamps,
    normalise,
)

# A window as the captions actually produce it: no punctuation, no casing,
# filler left in, and a couple of words missing.
TRANSCRIPT = (
    "so my my thesis on solana i think solana is the most performant uh and "
    "cheapest blockchain to do anything on which means that it s the best "
    "place for builders and retail and honestly it s just communism by "
    "another and i m killing two birds one stone streaming my trading on kick"
)

SOURCES = [{
    "title": "How Mizkif Made $20,000,000 Without Posting..",
    "text": TRANSCRIPT,
    "start_seconds": 2700.0,
    "timestamp": "45:00",
}]


def quotes_flagged(answer, sources=SOURCES):
    return [f.claim for f in check_quotes(answer, sources)]


class TestQuotesThatMustNotBeFlagged:
    """Correct behaviour that earlier versions called fabrication."""

    def test_verbatim_quote(self):
        answer = 'He said Solana is "the most performant and cheapest ' \
                 'blockchain to do anything on".'
        assert quotes_flagged(answer) == []

    def test_quote_dropping_a_spoken_filler(self):
        """The caption has "performant uh and cheapest"; nobody writes the uh."""
        answer = 'He called it "the most performant and cheapest blockchain".'
        assert quotes_flagged(answer) == []

    def test_quote_restoring_a_word_the_caption_lost(self):
        """Caption: "communism by another". He said "by another name"."""
        answer = 'He calls it "communism by another name".'
        assert quotes_flagged(answer) == []

    def test_quote_restoring_a_dropped_preposition(self):
        """Caption: "killing two birds one stone"."""
        answer = 'He describes "killing two birds with one stone" on stream.'
        assert quotes_flagged(answer) == []

    def test_episode_title_is_source_material(self):
        answer = 'The episode "How Mizkif Made $20,000,000 Without ' \
                 'Posting.." covers it.'
        assert quotes_flagged(answer) == []

    def test_ellipsis_marks_omitted_speech(self):
        answer = 'He said "solana is the most performant... blockchain to ' \
                 'do anything on".'
        assert quotes_flagged(answer) == []

    def test_bracketed_substitution(self):
        answer = 'He said "it s just [expletive] communism by another name".'
        assert quotes_flagged(answer) == []

    def test_two_quotations_in_one_answer(self):
        """The gap between them is prose, not a third quotation."""
        answer = ('He said "the most performant and cheapest blockchain" and '
                  'later called it "communism by another name".')
        assert quotes_flagged(answer) == []

    def test_scare_quotes_are_not_claims(self):
        assert extract_quotes('he called it a "scam" and a "rug"') == []


class TestFabricationsThatMustBeCaught:
    def test_invented_sentence(self):
        answer = 'Ansem said "bitcoin will reach four hundred thousand ' \
                 'dollars by next march".'
        assert quotes_flagged(answer)

    def test_invented_clause_hidden_behind_an_ellipsis(self):
        """Elision excuses what it replaces, never what surrounds it."""
        answer = ('He said "solana is the most performant... and bitcoin '
                  'will reach four hundred thousand by march".')
        assert quotes_flagged(answer)

    def test_words_scattered_across_the_transcript_do_not_add_up(self):
        """Locality is the whole defence against accumulating a false pass."""
        answer = 'He said "solana streaming builders communism stone kick".'
        assert quotes_flagged(answer)

    def test_paraphrase_presented_as_a_quotation(self):
        """The real finding from the first live run: a punchy rewording,
        in quotation marks, that the speaker never used."""
        answer = 'Ansem said they "fucked up their tokenomics".'
        assert quotes_flagged(answer)

    def test_quote_with_no_sources_at_all(self):
        answer = 'He said "the most performant and cheapest blockchain".'
        assert check_quotes(answer, [])


class TestAttribution:
    """Quotation marks do two jobs; only one of them is a claim.

    Both examples here are real support-bot output that the checker called
    fabricated, which would have blocked a release over a figure of speech.
    """

    def test_the_bots_own_illustration_is_not_a_fabricated_quote(self):
        answer = ('it fills at your price, instead of letting it fill at '
                  '"whatever the market gives you today".')
        findings = check_quotes(answer, SOURCES)
        assert findings, "still worth reporting"
        assert not any(f.hard for f in findings), "but must not block a release"

    def test_an_e_g_example_is_not_a_fabricated_quote(self):
        answer = ('good for time-bound plays (e.g., "only want this order '
                  'live until an event happens").')
        assert not any(f.hard for f in check_quotes(answer, SOURCES))

    def test_an_attributed_quote_still_blocks(self):
        answer = 'Ansem said "bitcoin will reach four hundred thousand by march".'
        assert any(f.hard for f in check_quotes(answer, SOURCES))

    def test_attribution_wins_over_a_nearby_hedge(self):
        """"called it" is illustrative on its own, but "he said he called it"
        is still an attribution."""
        answer = 'He said he called it "bitcoin reaching four hundred thousand".'
        assert check_quotes(answer, SOURCES)


class TestCoverage:
    def test_exact_containment_scores_full(self):
        assert coverage("communism by another", TRANSCRIPT) == 1.0

    def test_one_missing_word_leaves_exactly_one_unmatched(self):
        """75%, not 85% — which is why short quotes are judged on words
        unaccounted for rather than on a percentage."""
        assert coverage("communism by another name", TRANSCRIPT) == 0.75

    def test_unrelated_text_scores_low(self):
        assert coverage("ethereum gas fees are far too high", TRANSCRIPT) < 0.5

    def test_empty_inputs_do_not_raise(self):
        assert coverage("", TRANSCRIPT) == 0.0
        assert coverage("anything", "") == 0.0


class TestExtraction:
    @pytest.mark.parametrize("text,expected", [
        ("they won 3:1", []),                       # a score, not a time
        ("at 45:22 he says", [("45:22", 2722)]),
        ("around 2:13:41", [("2:13:41", 8021)]),
    ])
    def test_timestamps(self, text, expected):
        assert extract_timestamps(text) == expected

    def test_figures(self):
        found = extract_figures("it went from $1 to $200, up 45%")
        assert "$1" in found and "45%" in found

    def test_normalise_strips_caption_noise(self):
        """Filler, casing and punctuation go; ordinary words stay. "the" is
        not filler — dropping real words from both sides would let unrelated
        sentences match each other."""
        assert normalise("Uh, the MOST performant!") == "the most performant"


class TestTimestampsAndFigures:
    def test_a_cited_moment_near_a_returned_window_is_fine(self):
        findings = audit("Around 45:30 he explains it.", SOURCES)
        assert not [f for f in findings if f.kind == "timestamp"]

    def test_a_moment_far_from_every_window_is_reported(self):
        findings = audit("Around 2:13:41 he explains it.", SOURCES)
        assert [f for f in findings if f.kind == "timestamp"]

    def test_timestamp_and_figure_findings_are_never_blocking(self):
        """Only fabricated quotes gate a release; rounding a timestamp or
        restating a figure in other units is legitimate."""
        findings = audit("Around 2:13:41 the target was $5,000.", SOURCES)
        assert findings and not any(f.hard for f in findings)
