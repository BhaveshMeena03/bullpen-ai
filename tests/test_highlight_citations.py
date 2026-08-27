"""Every unprompted fact must point at the moment it came from.

A wrong answer is a bad answer. A confident citation to a moment that
says something else is worse: it is the one thing this archive exists to
be trusted about, and it is posted unprompted, to people who did not ask.

Two entries in the live pool were wrong. One read "Ansem said SpaceX
traded at 175 on Hyperliquid" and pointed at 3:07:00, where the show is
running a bracket on who is best looking. The real discussion is at
14:07, nearly three hours earlier. It went out in public under a
compliment, with a link that opened on the wrong segment.

This runs where the transcripts are — they are 7MB and deliberately not
in the image, so the check belongs here rather than at load time.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.x_bot import load_highlights  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"


def _maker():
    spec = importlib.util.spec_from_file_location(
        "make_highlights", ROOT / "scripts" / "make_highlights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not EPISODES.exists(), reason="transcripts not present")
def test_every_shipped_highlight_cites_its_own_moment():
    maker = _maker()
    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}

    wrong = []
    for entry in load_highlights():
        episode = episodes.get(entry.get("episode_id"))
        assert episode, f"highlight references a missing episode: {entry}"
        if not maker.cites_its_moment(episode, entry["timestamp"],
                                      entry["text"]):
            wrong.append(f"{entry['timestamp']} — {entry['text'][:70]}")

    assert not wrong, (
        "these highlights point at a moment that does not mention them, and "
        "they get posted unprompted with a link:\n  " + "\n  ".join(wrong)
        + "\n\nRerun: .venv/bin/python scripts/make_highlights.py --verify")


@pytest.mark.skipif(not EPISODES.exists(), reason="transcripts not present")
def test_the_check_catches_a_citation_that_is_hours_out():
    """Guards the check itself — a verifier that passes everything is
    worse than none, because it reads as coverage."""
    maker = _maker()
    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}
    episode = episodes["lTWv-SIEFpo"]

    # The real error, restored: the claim is at 14:07, cited at 3:07:00.
    claim = ("Ansem said SpaceX traded at 175 on Hyperliquid before opening "
             "at 150, then ran from 150 to 225.")
    assert not maker.cites_its_moment(episode, "3:07:00", claim)
    assert maker.cites_its_moment(episode, "14:07", claim)


def test_the_pool_is_not_empty():
    """Dropping bad entries must not quietly empty the pool — that would
    turn every compliment back into silence."""
    assert len(load_highlights()) >= 10
