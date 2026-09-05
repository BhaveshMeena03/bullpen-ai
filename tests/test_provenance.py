"""Both of these guards exist because the failure happened, not because it
was imagined. The tests carry the real cases."""
import pytest

from app.provenance import (drop_hallucinated, is_original_publisher,
                            looks_like_someone_elses_cut, admissible)


def _seg(t, text):
    return {"t": t, "text": text}


def test_the_hold_music_hallucination_is_removed():
    """Tesla's Q2 2026 call opens with 8.5 minutes of hold music, over
    which Whisper wrote "Let's get started." twenty-nine times."""
    junk = [_seg(i * 30, "Let's get started.") for i in range(29)]
    real = [_seg(600, "Thanks everyone for joining."),
            _seg(605, "We had a record quarter on deliveries.")]
    kept, notes = drop_hallucinated(junk + real)
    assert [s["text"] for s in kept] == [s["text"] for s in real]
    assert notes and "x29" in notes[0]


def test_a_person_agreeing_is_not_a_hallucination():
    """Measured on a real 33-minute interview: "Yeah." nine times, "Yes."
    four, "Thank you." four. Deleting those would delete speech."""
    segs = ([_seg(i, "Yeah.") for i in range(9)]
            + [_seg(100 + i, "Yes.") for i in range(4)]
            + [_seg(200, "Autonomy is the thing that matters here.")])
    kept, notes = drop_hallucinated(segs)
    assert len(kept) == len(segs)
    assert notes == []


def test_a_short_interjection_survives_even_when_very_repeated():
    """A one-word agreement fifteen times across an hour is a person."""
    segs = [_seg(i * 240, "Right.") for i in range(15)]
    kept, _ = drop_hallucinated(segs)
    assert len(kept) == 15


@pytest.mark.parametrize("channel,expected", [
    ("Lex Fridman", True),
    ("Tesla", True),
    ("PowerfulJRE", True),
    ("Game Time", False),            # the re-upload found in a real search
    ("Common Sense Skeptic", False),
    ("Brighter with Herbert", False),
    ("", False),
])
def test_only_the_original_publisher_is_admissible(channel, expected):
    assert is_original_publisher(channel) is expected


@pytest.mark.parametrize("title", [
    "DEBUNKING MUSK - TED Talk 2022 Pt 1/4",
    "Elon Musk Best Of Compilation",
    "Elon Musk motivational speech",
    "Lex Fridman #18 REUPLOAD",
])
def test_a_title_that_advertises_a_re_cut_is_refused(title):
    assert looks_like_someone_elses_cut(title) is True


def test_a_real_title_is_not_refused():
    assert looks_like_someone_elses_cut(
        "Elon Musk: Tesla Autopilot | Lex Fridman Podcast #18") is False


def test_the_game_time_reupload_is_refused_by_channel():
    """The exact pair a search returned: the original and a copy of the
    same 1,965-second file on someone else's channel."""
    ok, _ = admissible("Lex Fridman",
                       "Elon Musk: Tesla Autopilot | Lex Fridman Podcast #18")
    assert ok is True
    ok, why = admissible("Game Time",
                         "Elon Musk : Tesla Autopilot, Lex Fridman Podcast")
    assert ok is False and "Game Time" in why
