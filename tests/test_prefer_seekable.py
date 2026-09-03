"""Where the same moment exists twice, cite the copy you can jump to.

About half of every live broadcast is also inside the YouTube upload of
that episode, so one query can retrieve the same passage from both. They
are not equivalent: a YouTube citation carries ?t= and lands on the second
being quoted, while an X one cannot — X has no timestamp parameter for
video. Returning the X copy when the YouTube one exists costs the reader
the whole point of a citation.

The failure to guard against is the opposite one: dropping a passage that
only exists on X. Roughly half of each broadcast is not in the upload at
all, and that half is the reason any of this was built.
"""

from app.podcast import _can_seek, _prefer_seekable, _same_moment
from app.schemas import PodcastHit

YT = "https://www.youtube.com/watch?v=abc123&t=3407s"
X = "https://x.com/MarketBubble/status/2077854740575092752"

# The same speech as rendered by YouTube's auto-captions and by Whisper —
# different casing and punctuation, same words.
SAME_A = ("base is the blockchain for global finance the goal is to "
          "basically make it so trading payments agents run on this")
SAME_B = ("Base is the blockchain for global finance. The goal is to "
          "basically make it so trading, payments, agents run on this")
OTHER = ("completely different stretch about robotics and manufacturing "
         "capacity in china over the coming decade")


def hit(episode_id: str, link: str, text: str) -> PodcastHit:
    return PodcastHit(episode_id=episode_id, title="Episode", start_seconds=60,
                      timestamp="1:00", deep_link=link, text=text, score=0.6)


def test_recognises_which_links_can_seek():
    """X broadcasts seek too — ?t=<seconds> opens the player there.

    Asserted the other way for months on an assumption nobody tested, which
    cost half the archive its play button. Verified against three live
    broadcasts before this was changed.
    """
    assert _can_seek(YT)
    assert _can_seek(X)
    assert not _can_seek("")


def test_matches_the_same_moment_across_two_transcriptions():
    """The two sources never produce identical text, so an exact compare
    would never fire and this whole guard would be dead code."""
    assert _same_moment(SAME_A, SAME_B)
    assert not _same_moment(SAME_A, OTHER)


def test_drops_the_unjumpable_copy_when_youtube_has_it():
    kept = _prefer_seekable([hit("yt", YT, SAME_A), hit("x-1", X, SAME_B)])
    assert [h.episode_id for h in kept] == ["yt"]


def test_wins_even_when_the_x_copy_ranked_higher():
    """Reranking can put the X copy first; that must not decide it."""
    kept = _prefer_seekable([hit("x-1", X, SAME_B), hit("yt", YT, SAME_A)])
    assert [h.episode_id for h in kept] == ["yt"]


def test_never_drops_content_that_exists_only_on_x():
    """The half of each broadcast the upload cuts is the reason for all of
    this. Losing it to a dedupe would defeat the purpose entirely."""
    kept = _prefer_seekable([hit("x-1", X, OTHER)])
    assert [h.episode_id for h in kept] == ["x-1"]


def test_keeps_unrelated_x_material_alongside_youtube():
    kept = _prefer_seekable([hit("yt", YT, SAME_A), hit("x-1", X, OTHER)])
    assert [h.episode_id for h in kept] == ["yt", "x-1"]


def test_leaves_an_all_youtube_result_set_untouched():
    hits = [hit("a", YT, SAME_A), hit("b", YT, OTHER)]
    assert [h.episode_id for h in _prefer_seekable(hits)] == ["a", "b"]


def test_short_windows_are_never_treated_as_matching():
    """Two three-word windows share words by chance; treating that as the
    same moment would delete unrelated results."""
    assert not _same_moment("yeah exactly right", "yeah exactly right")
