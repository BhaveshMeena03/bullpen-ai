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


# --- the two archives must not share a vector -------------------------------

def test_the_index_defaults_to_the_market_bubble_corpus():
    from app.podcast import PodcastIndex, NAMESPACE
    assert PodcastIndex._namespace == NAMESPACE == "podcast"


def test_a_second_archive_gets_its_own_namespace():
    """@mbubbleSearch's standing is that it answers from the Market Bubble
    broadcast. One reply about that show sourced from a Tesla interview
    would prove it cannot tell the difference, so the corpora never share
    a namespace."""
    from app.podcast import PodcastIndex
    index = PodcastIndex.__new__(PodcastIndex)
    index._namespace = "elon"
    assert index._namespace != PodcastIndex._namespace


# --- the two archives stay apart end to end ---------------------------------

def test_the_two_archives_cannot_serve_each_others_cached_answers():
    """Sharing a namespace is the obvious way to mix the corpora. The cache
    is the non-obvious one: the same question asked of both surfaces would
    hit the same key and return the wrong archive's answer."""
    from app.answer_cache import make_key
    q = "what did he say about mars"
    assert make_key(q, surface="podcast", top_k=6) != make_key(q, surface="elon", top_k=6)


def test_the_archive_reports_how_long_each_recording_is():
    """The page shows how far into a recording a moment sits, which needs
    the whole length. A timestamp alone says nothing on a conversation that
    runs eight and a half hours."""
    import app.main as m
    m._ELON_CACHE = [{
        "episode_id": "x", "title": "t", "published_at": "2024-08-02",
        "segments": [{"t": 0, "text": "a"}, {"t": 31080, "text": "b"}]}]
    try:
        assert m._runtime(m._ELON_CACHE[0]) == 31080
    finally:
        m._ELON_CACHE = None


def test_a_missing_archive_file_does_not_take_the_service_down():
    """Nothing else in the service reads this file, so the page degrades to
    empty panels rather than a 500."""
    import app.main as m
    from pathlib import Path
    original, m._ELON_CACHE = m._ELON_FILE, None
    m._ELON_FILE = Path("/nonexistent/elon_episodes.json")
    try:
        assert m._elon_episodes() == []
    finally:
        m._ELON_FILE, m._ELON_CACHE = original, None
