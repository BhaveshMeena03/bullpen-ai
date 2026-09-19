"""A moment in the live broadcast plays on YouTube only when it is there.

The two copies run on different clocks: the upload drops the pre-show and
about a quarter of the rest, in pieces. So a second maps only between two
anchors that agree on the offset, and a passage only points at YouTube
when all of it does. Anything else stays on X, which also lands on the
second -- a wrong minute on YouTube would be worse than the right one on X.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import youtube_map  # noqa: E402

# A broadcast whose upload runs 110s behind until a cut at 600s, after
# which it runs 300s behind.
ANCHORS = {"x-live": [[t, "YT1", t - 110] for t in range(100, 600, 10)]
                     + [[t, "YT1", t - 300] for t in range(620, 1200, 10)]}


def _use(monkeypatch):
    monkeypatch.setattr(youtube_map, "_anchors", lambda: ANCHORS)


def test_maps_a_second_between_agreeing_anchors(monkeypatch):
    _use(monkeypatch)
    assert youtube_map.youtube_moment("x-live", 305) == ("YT1", 195)


def test_refuses_a_second_across_a_cut(monkeypatch):
    """590 and 620 disagree by 190 seconds: the upload skips between them."""
    _use(monkeypatch)
    assert youtube_map.youtube_moment("x-live", 605) is None


def test_refuses_outside_the_anchors_and_for_unknown_episodes(monkeypatch):
    _use(monkeypatch)
    assert youtube_map.youtube_moment("x-live", 50) is None
    assert youtube_map.youtube_moment("x-other", 305) is None


def test_a_passage_points_at_youtube_only_when_all_of_it_is_there(monkeypatch):
    _use(monkeypatch)
    whole = youtube_map.youtube_first(
        {"episode_id": "x-live", "start_seconds": 200, "deep_link": "https://x.com/a"})
    assert whole["deep_link"] == "https://www.youtube.com/watch?v=YT1&t=90s"
    assert whole["youtube_offset"] == -110
    assert whole["live_link"] == "https://x.com/a"
    # Starts before the cut and runs across it: the end would land wrong.
    split = {"episode_id": "x-live", "start_seconds": 520, "deep_link": "https://x.com/b"}
    assert youtube_map.youtube_first(split) == split


def test_youtube_passages_are_left_alone(monkeypatch):
    _use(monkeypatch)
    hit = {"episode_id": "abcDEF12345", "start_seconds": 10, "deep_link": "https://youtube.com/x"}
    assert youtube_map.youtube_first(hit) is hit


def test_summary_timestamps_map_where_they_can(monkeypatch):
    _use(monkeypatch)
    got = youtube_map.summary_moments("x-live", "[0:05:05] one [0:10:05] two [0:30:00] three")
    assert got == {"305": ["YT1", 195]}
