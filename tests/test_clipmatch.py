"""Placing a clip, and refusing to place one that is not ours.

A clipper posts thirty seconds and captions it "must watch". The bot had
nothing to read, fell through to the newest episode, and answered with
3,890 characters about it -- correct only when the clip happened to come
from that show.

Both halves are tested here, and the refusal is the half that matters.
A matcher that always answers is worse than none: it turns "I don't know
which show this is" into a confident paragraph about the wrong one.

The positive fixture is real. It is the archive's own words from Market
Bubble ep 19 around 5:12, which is where a live clipper's post was placed
on 14 Sep 2026 -- 138 unique runs, share_in_span 1.00, runner_up 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app import clipmatch  # noqa: E402


def episode(eid: str, title: str, lines: list[tuple[float, str]]) -> dict:
    return {"episode_id": eid, "title": title,
            "segments": [{"t": t, "text": text} for t, text in lines]}


# Ep 19 around 5:04-5:35, close to verbatim.
REAL = episode("x-19", "HUNTER BIDEN: Market Bubble Episode 19", [
    (304.0, "a coin at a trillion dollars"),
    (306.0, "and then people are just like immediately selling it"),
    (308.0, "it's so stupid it's crazy"),
    (312.0, "the insiders benefited a lot from the launch"),
    (315.0, "no matter how they say they structured the token supply"),
    (317.0, "but how much do you think how much"),
    (321.0, "because I generally don't know I have no idea"),
    (325.0, "it's safe to say people that we've never heard of"),
    (328.0, "that are kind of behind the scenes have made millions"),
    (331.0, "and millions of dollars on this launch"),
])

OTHER = episode("x-11", "LIVE W/ JESSE POLLAK & LUCAS BRUDER", [
    (100.0, "the average person today is broke and unhappy"),
    (104.0, "so just go the other way and do the opposite"),
    (108.0, "my parents my best friends everyone I told this to"),
    (112.0, "made me feel bad about it and shit on me for it"),
])

ARCHIVE = [REAL, OTHER]


class TestItPlacesAClipFromTheArchive:
    def test_a_clip_is_placed_on_the_right_episode(self):
        """Whisper's wording differs from the archive's -- it drops a
        phrase here and merges another -- which is exactly why runs are
        matched rather than lines."""
        clip = ("the insiders benefited a lot from the launch no matter "
                "how they say they structured the token supply but how "
                "much do you think how much because I generally don't "
                "know I have no idea")
        got = clipmatch.place(clip, ARCHIVE)
        assert got is not None, "refused a clip that is in the archive"
        assert got["episode_id"] == "x-19"
        assert got["matches"] >= clipmatch.MIN_UNIQUE_RUNS

    def test_it_reports_where_in_the_episode(self):
        """The span is the point. Answering needs the moment, not the
        episode -- the value is the transcript AFTER the clip ends."""
        clip = ("the insiders benefited a lot from the launch no matter "
                "how they say they structured the token supply but how "
                "much do you think how much because I generally don't know")
        got = clipmatch.place(clip, ARCHIVE)
        assert 300.0 <= got["start"] <= 320.0
        assert got["share_in_span"] >= clipmatch.MIN_IN_SPAN

    def test_the_wrong_episode_is_not_chosen(self):
        clip = ("the average person today is broke and unhappy so just go "
                "the other way and do the opposite my parents my best "
                "friends everyone I told this to")
        got = clipmatch.place(clip, ARCHIVE)
        assert got is not None and got["episode_id"] == "x-11"


class TestItRefusesWhatItCannotPlace:
    """The half that keeps the bot honest. Each of these returned None
    against the real 39-episode archive before being written down."""

    @pytest.mark.parametrize("label,text", [
        ("a different domain entirely",
         "we the people of the united states in order to form a more "
         "perfect union establish justice insure domestic tranquility "
         "provide for the common defence promote the general welfare"),
        ("a cooking show",
         "today we are making a classic beef bourguignon start by browning "
         "the bacon lardons in a heavy casserole then remove them and sear "
         "the beef in batches until deeply coloured on every side"),
        # The dangerous one: the show's own register, none of its words.
        ("generic crypto chatter",
         "yeah i think the market is going to be interesting this week you "
         "know like people are just going to keep buying and selling and "
         "that is how it works right so we will see what happens next"),
    ])
    def test_a_clip_from_elsewhere_is_refused(self, label, text):
        assert clipmatch.place(text, ARCHIVE) is None, label

    def test_a_clip_too_short_to_judge_is_refused(self):
        """Fewer runs than the floor is not a weak match, it is no
        evidence. Six words cannot pin four hours of tape."""
        assert clipmatch.place("i think the market is going up", ARCHIVE) is None

    def test_an_empty_transcript_is_refused(self):
        assert clipmatch.place("", ARCHIVE) is None
        assert clipmatch.place("   ", ARCHIVE) is None

    def test_an_empty_archive_refuses_rather_than_raises(self):
        clip = ("the insiders benefited a lot from the launch no matter how "
                "they say they structured the token supply but how much")
        assert clipmatch.place(clip, []) is None


class TestTheUniquenessRule:
    def test_a_phrase_repeated_in_the_episode_is_not_evidence(self):
        """align_recordings.py's rule, kept: a run appearing twice in the
        episode would pair the clip's one instance with whichever it met
        first and invent a location. Only runs unique in the episode count.
        """
        repeated = episode("x-r", "Repeats", [
            (10.0, "one two three four five six seven eight nine"),
            (200.0, "one two three four five six seven eight nine"),
            (400.0, "one two three four five six seven eight nine"),
        ])
        clip = "one two three four five six seven eight nine"
        assert clipmatch.place(clip, [repeated]) is None


# --- the span gate, which was dead code for its first hour -----------
#
# _densest() originally computed its own window from the matches it was
# judging:
#
#     hi = anchor + (max(order) - min(order)) + SPAN_SLACK_SECONDS
#
# That is the spread of those very matches, so the window always covered
# all of them and the share was 1.00 by construction. MIN_IN_SPAN
# rejected nothing, and a mutation run setting it to 0.0 changed no
# result -- which is how the dead path was found, since a threshold that
# can be deleted without breaking a test is not being tested.
#
# Caught by stitching three chunks from points four hours apart in one
# episode: a perfect 1.00, placed as though it were one continuous clip.
# A window that judges the data cannot be derived from the data.

SPREAD = episode("x-spread", "Four hours of tape", [
    (10.0, "the insiders benefited a lot from the launch"),
    (14.0, "no matter how they say they structured the token supply"),
    (3600.0, "it's safe to say people that we've never heard of"),
    (3604.0, "that are kind of behind the scenes have made millions"),
    (7200.0, "because I generally don't know I have no idea"),
    (7204.0, "but how much do you think how much"),
])


class TestTheSpanGate:
    def test_matches_scattered_across_hours_are_refused(self):
        """A real clip is contiguous. Runs landing two hours apart are
        coincidence, however many there are."""
        clip = ("the insiders benefited a lot from the launch no matter "
                "how they say they structured the token supply "
                "it's safe to say people that we've never heard of "
                "that are kind of behind the scenes have made millions "
                "because I generally don't know I have no idea "
                "but how much do you think how much")
        assert clipmatch.place(clip, [SPREAD], duration_seconds=60.0) is None

    def test_the_same_runs_close_together_are_placed(self):
        """The control for the test above: identical text, one stretch of
        tape. If this failed, the gate would be rejecting real clips."""
        tight = episode("x-tight", "One minute of tape", [
            (10.0, "the insiders benefited a lot from the launch"),
            (14.0, "no matter how they say they structured the token supply"),
            (20.0, "it's safe to say people that we've never heard of"),
            (24.0, "that are kind of behind the scenes have made millions"),
            (30.0, "because I generally don't know I have no idea"),
            (34.0, "but how much do you think how much"),
        ])
        clip = ("the insiders benefited a lot from the launch no matter "
                "how they say they structured the token supply "
                "it's safe to say people that we've never heard of "
                "that are kind of behind the scenes have made millions "
                "because I generally don't know I have no idea "
                "but how much do you think how much")
        got = clipmatch.place(clip, [tight], duration_seconds=60.0)
        assert got is not None and got["episode_id"] == "x-tight"

    def test_a_known_duration_narrows_the_window(self):
        """A thirty-second clip cannot span two hours, and saying so is
        what the caller's duration_ms buys."""
        clip = ("the insiders benefited a lot from the launch no matter "
                "how they say they structured the token supply "
                "it's safe to say people that we've never heard of "
                "that are kind of behind the scenes have made millions")
        assert clipmatch.place(clip, [SPREAD], duration_seconds=30.0) is None


class TestTheEvidenceFloor:
    """MIN_UNIQUE_RUNS, tested by its VALUE rather than by accident.

    test_a_clip_too_short_to_judge_is_refused above uses a six-word
    string, which yields no eight-word runs at all -- so it is caught by
    the empty-list guard and passes even with the floor set to 1. A
    mutation run proved that: dropping MIN_UNIQUE_RUNS from 12 to 1 broke
    nothing, which meant the threshold was not under test.

    These two sit either side of the boundary, both drawn from text that
    genuinely matches REAL, so the only thing separating them is how much
    evidence there is.
    """

    # 18 words -> 11 unique matches, one short of the floor.
    NEARLY = ("the insiders benefited a lot from the launch no matter how "
              "they say they structured the token supply")
    # 20 words -> 13 unique matches, one over.
    ENOUGH = ("the insiders benefited a lot from the launch no matter how "
              "they say they structured the token supply but how much")

    def test_just_under_the_floor_is_refused(self):
        """Eleven matching runs is a real overlap and still not enough to
        pin four hours of tape. Refusing here is the whole design: a weak
        match answered confidently is the failure this module exists to
        prevent."""
        assert clipmatch.place(self.NEARLY, ARCHIVE,
                               duration_seconds=30.0) is None

    def test_just_over_the_floor_is_placed(self):
        """The control. If this ever fails the floor has been raised past
        what a short clip can supply, and real clips start being refused."""
        got = clipmatch.place(self.ENOUGH, ARCHIVE, duration_seconds=30.0)
        assert got is not None and got["episode_id"] == "x-19"
