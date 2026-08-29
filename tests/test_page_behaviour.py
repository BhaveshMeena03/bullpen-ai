"""Two promises the page makes, guarded at the source.

Neither is checkable from Python at runtime — they live in the browser —
so this reads the page and asserts the shapes are still there. It is a
cheap guard against the two regressions that would be invisible in the
test suite and obvious to a visitor.

A timestamp inside an episode summary used to be a link that opened
YouTube in a new tab, which is exactly what the rest of the page stopped
doing when citations learned to play in place. An X broadcast keeps the
link, because X still cannot jump to a time in video and a button that
opened it at 0:00 would promise more than a plain link does.

And the note explaining that X cannot jump used to sit under the results
heading, so a set where four of six results played announced to all six
that nothing would. It belongs on the cards it is true of.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGE = (ROOT / "demo" / "podcast.html").read_text()


def test_a_summary_timestamp_plays_rather_than_navigating_away():
    assert 'class="cite ep-cite"' in PAGE
    assert 'class="player ep-player"' in PAGE
    assert 'querySelector(".ep-player")' in PAGE


def test_a_broadcast_summary_keeps_its_outbound_link():
    """canSeek gates the two shapes. Without the gate, an X timestamp
    would render a button that cannot do anything."""
    assert "var seekable = canSeek(url)" in PAGE
    assert "if(!seekable){" in PAGE


def test_the_scrub_note_is_not_attached_to_the_heading():
    """It reads as a property of the result set when it is a property of
    one result."""
    assert "'<h3>'+head+'</h3>';" in PAGE
    assert "'<h3>'+head+'</h3>'+note" not in PAGE


def test_the_scrub_note_sits_where_the_play_button_would_be():
    """Same row, so it reads as the reason that card has no play button
    rather than as an unrelated aside."""
    actions = PAGE[PAGE.index('class="hit-actions"'):]
    play = actions.index("play here")
    note = actions.index('class="seeknote"')
    assert note - play < 400, "the note drifted out of the action row"


def test_only_one_player_can_be_open():
    """Two videos playing at once is the worst failure this page has, and
    the reset runs across every slot, not only the ones nearby."""
    assert PAGE.count('querySelectorAll(".player").forEach') >= 2
