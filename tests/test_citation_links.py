"""The link has to point at the moment the answer quotes.

Both cases here are real replies, found by previewing questions before
posting them.

Asked about Solana reclaiming $100, the model correctly quoted Ep 10 at
3:36:29 — "I'm expecting Solana to go to 150 during Q3" — and the reply
linked the August 27 fantasy football draft, because that was the
top-ranked passage. The sentence was true and the link landed three hours
into an unrelated show.

Asked what Brian and Banks talked about, the answer said Armstrong was
"scheduled to appear around 3:30 PM" and the reply offered to jump to
3:30 — three and a half minutes into a May episode. A clock time is not a
position in a recording, and the two are indistinguishable until the PM.

A wrong answer is a bad answer. A real timestamp welded to the wrong
episode is worse, because it looks checkable and is not, and checkable is
the only thing this tool sells.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.x_bot import _CITES_A_TIME, cited_hit, format_reply  # noqa: E402


class Hit:
    def __init__(self, episode_id, timestamp, deep_link, stamps=(), title="t"):
        self.episode_id = episode_id
        self.title = title
        self.timestamp = timestamp
        self.deep_link = deep_link
        self.text = "…"
        # What the model is actually shown: every line with its own stamp.
        self.text_ts = "\n".join(f"[{s}] line" for s in stamps)


FANTASY = Hit("x-2093074127011909791", "0:00",
              "https://x.com/MarketBubble/status/2093074127011909791",
              stamps=("0:00", "2:02", "4:24"))
EP10 = Hit("x-2075316750439338088", "3:34:11",
           "https://x.com/MarketBubble/status/2075316750439338088",
           stamps=("3:34:11", "3:36:29", "3:36:35"))


def test_the_link_follows_the_citation_not_the_ranking():
    """The real failure: 3:36:29 is Ep 10's, and the top hit was the
    August 27 draft."""
    hit, moment = cited_hit("Around 3:36:29 he expected Solana at 150.",
                            [FANTASY, EP10])
    assert hit is EP10
    assert moment == "3:36:29"


def test_the_reply_carries_the_right_episode_url():
    reply = format_reply("Around 3:36:29 he expected Solana at 150.",
                         [FANTASY, EP10], include_links=True, limit=1500)
    assert "2075316750439338088" in reply, "linked the wrong episode"
    assert "2093074127011909791" not in reply


def test_the_top_hit_is_used_when_it_is_the_cited_one():
    hit, moment = cited_hit("Around 2:02 they joked about the draft.",
                            [FANTASY, EP10])
    assert hit is FANTASY
    assert moment == "2:02"


def test_an_uncited_answer_keeps_the_best_ranked_passage():
    hit, moment = cited_hit("They discussed the draft at length.",
                            [FANTASY, EP10])
    assert hit is FANTASY
    assert moment is None


def test_a_moment_in_no_passage_does_not_move_the_link():
    """Naming a second that appears in nothing it was given is not a
    reason to build a link around that second."""
    hit, moment = cited_hit("Around 9:99:99 something happened.",
                            [FANTASY, EP10])
    assert moment is None


def test_passages_without_line_stamps_behave_as_before():
    """text_ts is empty on vectors written before it existed. With
    nothing to check against, absence of evidence is not evidence of a
    bad citation — dropping the link would be a regression."""
    old = Hit("old", "1:00", "https://youtube.com/watch?v=x&t=60s", stamps=())
    hit, moment = cited_hit("Around 7:02 he said it.", [old])
    assert hit is old
    assert moment == "7:02"


def test_a_clock_time_is_not_a_citation():
    """"scheduled to appear around 3:30 PM" offered a jump to 3:30."""
    assert not _CITES_A_TIME.search("scheduled to appear around 3:30 PM")
    assert not _CITES_A_TIME.search("he joined around 4:00 p.m.")


def test_a_real_timestamp_still_reads_as_one():
    for text, want in (("Around 2:28:13 in the episode", "2:28:13"),
                       ("at 9:51 he said", "9:51"),
                       ("around 12:14 they said", "12:14")):
        found = _CITES_A_TIME.search(text)
        assert found and found.group(0) == want


def test_a_clock_time_does_not_move_the_link():
    hit, moment = cited_hit("He was scheduled to appear around 3:30 PM.",
                            [FANTASY, EP10])
    assert moment is None, "a clock time was treated as a position"
