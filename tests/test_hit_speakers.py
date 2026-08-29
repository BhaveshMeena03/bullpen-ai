"""Who spoke, carried from the index to the page.

The speaker labels are written into Pinecone metadata by a script and
read back by a search. Nothing in between validates them, and the index
will return whatever was stored — including whatever a half-finished run
left behind. A malformed row must cost a chip on a card, never the
search.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas import PodcastHit  # noqa: E402


def hit(**over) -> PodcastHit:
    base = dict(episode_id="e1", title="t", start_seconds=0.0,
                timestamp="0:00", deep_link="https://x", text="x",
                score=0.5)
    base.update(over)
    return PodcastHit(**base)


def clean(raw) -> list[str]:
    """The read that app/podcast.py performs on the metadata."""
    return [str(x) for x in (raw or []) if isinstance(x, str)]


def test_a_hit_defaults_to_no_speakers():
    """Absent means "not established", never "nobody spoke" — only the
    hosts are labelled, so most passages have none."""
    assert hit().speakers == []


def test_speakers_survive_the_round_trip():
    assert hit(speakers=["Ansem"]).speakers == ["Ansem"]
    assert hit(speakers=["FaZe Banks", "Ansem"]).speakers == [
        "FaZe Banks", "Ansem"]


def test_metadata_that_is_missing_reads_as_empty():
    assert clean(None) == []
    assert clean([]) == []


def test_metadata_that_is_the_wrong_shape_does_not_raise():
    """Pinecone hands back what was stored. A number, a nested list or a
    None inside the list must not reach the page or take the search with
    it."""
    assert clean(["Ansem", None, 7, ["nested"]]) == ["Ansem"]
    assert clean([1, 2, 3]) == []


def test_the_field_reaches_a_serialized_response():
    """text_ts is excluded from API responses on purpose; speakers is
    not, because the card is meant to show it."""
    payload = hit(speakers=["Ansem"]).model_dump()
    assert payload["speakers"] == ["Ansem"]
    assert "text_ts" not in payload
