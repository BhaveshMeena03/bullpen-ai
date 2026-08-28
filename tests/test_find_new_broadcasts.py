"""What counts as a broadcast worth indexing.

The account posts three things that all carry video: the full three-to-four
hour show, the 30-60 minute cut-downs it puts out between shows, and short
teasers. Only the first is worth indexing — a cut-down is already inside
the full episode, so indexing it would duplicate passages and split the
citations for one moment across two sources.

And the show's post goes up when the stream STARTS. Fetched an hour in you
get the first hour, stored as if it were the episode, with nothing that
ever re-checks it.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.find_new_broadcasts import new_broadcasts  # noqa: E402

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
LONG_AGO = (NOW - timedelta(days=2)).isoformat().replace("+00:00", "Z")

FULL_SHOW = 4 * 3_600_000
CUT_DOWN = 45 * 60_000


def post(pid="1", text="LIVE W/ LUCA NETZ: Market Bubble Ep 10",
         created=LONG_AGO, keys=("k1",), **extra):
    return {"id": pid, "text": text, "created_at": created,
            "attachments": {"media_keys": list(keys)}, **extra}


def video(duration=FULL_SHOW, key="k1"):
    return {key: {"media_key": key, "type": "video", "duration_ms": duration}}


def run(posts, media, indexed=frozenset(), **kw):
    kw.setdefault("now", NOW)
    return new_broadcasts(posts, media, set(indexed), **kw)


def test_a_full_show_is_found():
    found, _ = run([post()], video())
    assert [p["id"] for p, _ in found] == ["1"]


def test_a_cut_down_is_skipped():
    """45 minutes of a show already indexed in full."""
    found, _ = run([post()], video(CUT_DOWN))
    assert found == []


def test_the_threshold_is_adjustable():
    found, _ = run([post()], video(CUT_DOWN), min_hours=0.5)
    assert len(found) == 1


def test_an_unknown_length_is_kept_not_dropped():
    """X does not always report duration. Losing a whole show because a
    field was missing is worse than one line of manual review."""
    found, _ = run([post()], {"k1": {"media_key": "k1", "type": "video"}})
    assert len(found) == 1
    assert found[0][1] == 0


def test_a_live_broadcast_is_not_fetched_yet():
    """Posted an hour ago: the show is still running."""
    hour_ago = (NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    found, waiting = run([post(created=hour_ago)], video())
    assert found == []
    assert waiting == 1


def test_it_is_fetched_once_the_stream_has_ended():
    ended = (NOW - timedelta(hours=7)).isoformat().replace("+00:00", "Z")
    found, waiting = run([post(created=ended)], video())
    assert len(found) == 1
    assert waiting == 0


def test_the_settle_window_is_adjustable():
    two_hours = (NOW - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    found, _ = run([post(created=two_hours)], video(), settle_hours=1)
    assert len(found) == 1


def test_an_already_indexed_show_is_not_offered_again():
    found, _ = run([post(pid="55")], video(), indexed={"x-55"})
    assert found == []


def test_a_reply_is_not_a_broadcast():
    reply = post(referenced_tweets=[{"type": "replied_to", "id": "9"}])
    assert run([reply], video())[0] == []


def test_a_quote_is_not_a_broadcast():
    quote = post(referenced_tweets=[{"type": "quoted", "id": "9"}])
    assert run([quote], video())[0] == []


def test_a_post_with_no_video_is_skipped():
    assert run([post(keys=())], {})[0] == []


def test_ordinary_commentary_is_skipped():
    """Video plus chatter is a clip, not an episode."""
    assert run([post(text="gm. big day today")], video())[0] == []


def test_since_excludes_older_shows():
    assert run([post()], video(), since="2026-12-01")[0] == []
