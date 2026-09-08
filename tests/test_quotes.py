"""The verifier has to be wrong in the safe direction.

Two errors are possible and they are not equally bad. Calling an invented
quote real is the one that ends the account; calling a real quote invented
is the one that makes it useless. Both are tested here, and the fixtures
are built to look like actual transcripts rather than clean prose —
interjections in the middle, filler, no punctuation to lean on.
"""

from __future__ import annotations

from app.quotes import MIN_WORDS, check


def episode(episode_id: str, lines: list[tuple[float, str]]) -> dict:
    return {"episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "url": f"https://www.youtube.com/watch?v={episode_id}",
            "segments": [{"t": t, "text": text} for t, text in lines]}


ARCHIVE = [
    episode("ep1", [
        (0.0, "Welcome back to the show, we are talking about markets."),
        (12.0, "When do you think SpaceX will land a human being on Mars?"),
        # The interjection that broke the first version of the matcher: a
        # real quote is quoted without it, and the run is cut in half.
        (18.0, "Hmm."),
        (19.0, "Best case is about five years, worst case ten years."),
        (26.0, "What are the determining factors, would you say?"),
    ]),
    episode("ep2", [
        (0.0, "Totally different conversation about liquidity and funding."),
        (9.0, "There is a lot of noise before the end of this quarter."),
    ]),
]

SPEAKERS = {"ep1": {"1": "Interviewer", "3": "Elon Musk"}}


class TestItFindsWhatWasSaid:
    def test_a_verbatim_quote_is_found(self):
        r = check("Best case is about five years, worst case ten years.",
                  ARCHIVE)
        assert r["verdict"] == "found"
        assert r["episode_id"] == "ep1"
        assert r["seconds"] == 19.0

    def test_an_interjection_does_not_break_a_real_quote(self):
        """The transcript has "Hmm." in the middle; the quote does not.

        This is the ordinary shape of a true quote, not an edge case, and
        the contiguous-run version scored it 0.55 and called it not_found.
        """
        r = check("When do you think SpaceX will land a human being on "
                  "Mars? Best case is about five years, worst case ten "
                  "years.", ARCHIVE)
        assert r["verdict"] == "found"
        assert r["matched_share"] > 0.9
        assert r["longest_run"] < r["matched_words"], "matched across the gap"

    def test_punctuation_and_case_do_not_matter(self):
        """Punctuation is the transcriber's guess, not the speaker's."""
        r = check("BEST CASE IS ABOUT FIVE YEARS -- WORST CASE, TEN YEARS!!",
                  ARCHIVE)
        assert r["verdict"] == "found"

    def test_it_names_the_speaker_when_a_human_labelled_that_line(self):
        r = check("Best case is about five years, worst case ten years.",
                  ARCHIVE, SPEAKERS)
        assert r["speaker"] == "Elon Musk"

    def test_no_speaker_is_reported_rather_than_guessed(self):
        r = check("There is a lot of noise before the end of this quarter.",
                  ARCHIVE, SPEAKERS)
        assert r["speaker"] is None, "ep2 has no labels; do not invent one"


class TestItRefusesWhatWasNot:
    def test_an_invented_quote_is_not_found(self):
        r = check("I have decided to buy every remaining bitcoin and "
                  "delete the ethereum blockchain forever.", ARCHIVE)
        assert r["verdict"] == "not_found"

    def test_a_real_quote_with_an_invented_ending_is_partial(self):
        """The misquote that actually circulates: true line, false tail.

        Reported as partial rather than found, and the matched_text shows
        which half was real — which is the useful answer, and the reason
        the partial band is set low enough to catch this.
        """
        r = check("Best case is about five years, worst case ten years, and "
                  "I guarantee we will be there by 2027 with a million "
                  "people living there.", ARCHIVE)
        assert r["verdict"] == "partial"
        assert "worst case ten years" in r["matched_text"]
        assert "guarantee" not in r["matched_text"]

    def test_words_scattered_across_the_archive_do_not_add_up(self):
        """Half from ep1, half from ep2, and it must not stitch them.

        Without the locality window a claim could be assembled out of
        common phrases from unrelated hours and score as verified.
        """
        r = check("Best case is about five years there is a lot of noise "
                  "before the end of this quarter", ARCHIVE)
        assert r["verdict"] != "found"

    def test_a_short_phrase_is_refused_rather_than_confirmed(self):
        r = check("about five years", ARCHIVE)
        assert r["verdict"] == "too_short"
        assert str(MIN_WORDS) in r["detail"]


class TestItDoesNotOverstateItself:
    def test_every_answer_says_what_it_checked(self):
        for claim in ("Best case is about five years, worst case ten years.",
                      "Something nobody in this archive has ever uttered "
                      "at any point in time."):
            assert "not proof" in check(claim, ARCHIVE)["coverage"]

    def test_an_empty_archive_finds_nothing(self):
        r = check("Best case is about five years, worst case ten years.", [])
        assert r["verdict"] == "not_found"
