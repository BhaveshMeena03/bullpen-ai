"""What has to happen when an episode arrives, and used not to.

Market Bubble #17 went up on a Saturday. The sync ran that morning at
ten, hours before the upload, and was next scheduled for the following
Friday -- so the newest episode would have been invisible for six days
while the account told people it was not indexed.

Two failures underneath that, both of which would have repeated:

The show was ALREADY there. It had gone out as a live broadcast two days
earlier under the title "$100K POLYMARKET FANTASY FOOTBALL DRAFT NIGHT",
transcribed in full and searchable. Nobody could reach it by number,
because the number is read out of the title and that title had none.

And an episode ingested without the speaker pipeline is anonymous. It is
searchable, so nothing looks broken, but "what did Banks say" degrades to
"one of the hosts" on the one episode people are asking about.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.x_bot import _same_show_as, episode_number  # noqa: E402


def row(title: str, published: str, length: int = 100) -> dict:
    return {"title": title, "published_at": published, "summary": "x" * length}


BROADCAST = row("$100K POLYMARKET FANTASY FOOTBALL DRAFT NIGHT",
                "2026-08-27T00:00:00Z", 3000)
UPLOAD = row("Polymarket Fantasy Football Draft | Market Bubble #17",
             "2026-08-29T00:00:00Z", 1200)
OLDER = row("Why Streaming Is The New Meta | Market Bubble #15",
            "2026-08-14T00:00:00Z")
UNRELATED = row("Our full conversation with somebody", "2026-06-01T00:00:00Z")


class TestReadingTheNumber:
    def test_it_reads_a_number_from_a_title_that_has_one(self):
        assert episode_number(UPLOAD["title"]) == 17
        assert episode_number("LIVE W/ TJR & Mert Market Bubble EP 8") == 8

    def test_a_special_title_carries_no_number(self):
        """Which is the whole problem: this is a real episode."""
        assert episode_number(BROADCAST["title"]) is None


class TestFindingTheSameShow:
    def test_an_unnumbered_broadcast_is_found_by_its_date(self):
        found = _same_show_as([UPLOAD], [UPLOAD, BROADCAST, OLDER, UNRELATED])
        assert BROADCAST in found

    def test_a_distant_show_is_not_swept_in(self):
        found = _same_show_as([UPLOAD], [UPLOAD, UNRELATED])
        assert UNRELATED not in found

    def test_a_numbered_cut_is_never_added_twice(self):
        """It is already in `matches`; adding it again would let a short
        cut outvote the full broadcast on length."""
        found = _same_show_as([UPLOAD], [UPLOAD, OLDER])
        assert OLDER not in found and UPLOAD not in found

    def test_nothing_numbered_means_nothing_to_anchor_to(self):
        assert _same_show_as([], [BROADCAST]) == []

    def test_a_missing_or_unparseable_date_is_survived(self):
        broken = row("$100K SOMETHING", "not-a-date")
        assert _same_show_as([UPLOAD], [broken]) == []
        assert _same_show_as([row("Market Bubble #9", "")], [BROADCAST]) == []

    def test_the_longer_cut_still_wins(self):
        """_summary_for picks the longest, which is the point: the
        broadcast runs about twice the upload and contains everything."""
        both = [UPLOAD] + _same_show_as([UPLOAD], [UPLOAD, BROADCAST])
        assert max(both, key=lambda s: len(s["summary"])) is BROADCAST


class TestTheSyncLabelsSpeakers:
    """Source-level, because the real thing downloads two hours of audio
    and loads a voice encoder."""

    source = (ROOT / "scripts" / "sync_latest.py").read_text()

    def test_it_runs_the_speaker_pipeline(self):
        for step in ("fetch_audio.py", "label_speakers.py",
                     "build_speaker_map.py", "apply_speaker_labels.py"):
            assert step in self.source, step

    def test_it_labels_before_it_summarizes(self):
        """The summary names people, so it has to come second.

        Scoped to the new-episode block. Comparing first occurrences
        across the whole file fails on the repair path, which summarizes
        earlier and legitimately.
        """
        block = self.source[self.source.index("# 3. work out who is speaking"):]
        assert (block.index("label_speakers(vid, log)")
                < block.index("summaries.summarize"))

    def test_the_summary_is_given_the_speakers(self):
        assert "summarize(episode, speakers=speakers)" in self.source

    def test_the_repair_path_gets_them_too(self):
        """Backfilling a summary without names writes an anonymous one
        over an episode whose speakers are already known, and nothing
        revisits it."""
        assert "speakers=stored_speakers(vid)" in self.source

    def test_a_speaker_failure_does_not_lose_the_episode(self):
        """An unlabelled episode beats a failed sync. Every other step
        here is wrapped the same way."""
        assert "speaker {what} FAILED" in self.source
        assert "return None" in self.source
