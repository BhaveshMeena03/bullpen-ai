"""The page shows every passage the answer was written from.

The Elon and MCG payloads kept the first six hits while the model read all
of them -- retrieval adds exact-name matches on top of the top six, up to
eleven in practice. An answer could then cite a moment the page never
showed: measured against the live MCG archive, 12 answers in 37 did, and a
reader clicking the cited second found no passage for it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import main  # noqa: E402
from app.schemas import PodcastHit  # noqa: E402


def _hits(n: int) -> list[PodcastHit]:
    return [PodcastHit(episode_id=f"ep{i}", title=f"Episode {i}",
                       start_seconds=60.0 * i, timestamp=f"{i}:00",
                       deep_link=f"https://example.com/{i}", text=f"passage {i}",
                       score=1.0 - i / 100)
            for i in range(n)]


def test_mcg_payload_keeps_every_passage():
    result = SimpleNamespace(answer="cites 9:00", hits=_hits(9))
    assert len(main._mcg_payload("q", result)["hits"]) == 9


def test_elon_payload_keeps_every_passage():
    result = SimpleNamespace(answer="cites 9:00", hits=_hits(9))
    assert len(main._elon_payload("q", result)["hits"]) == 9
