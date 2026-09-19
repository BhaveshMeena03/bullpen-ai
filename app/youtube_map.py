"""Where a moment in a live X broadcast plays on YouTube, if it does.

YouTube plays inside the page; X opens another site. So a moment that is
in both copies should link to YouTube, at the right second on YouTube's
own clock, which differs from the broadcast's because the upload is cut
from it. data/youtube_map.json holds anchors -- the same eight words said
at a known second in each copy -- built by scripts/build_youtube_map.py.

The rule is written to refuse rather than guess. A second maps only when
the anchors on both sides of it agree on the offset: then there is no cut
between them, and the moment is on YouTube where the offset says. Near
the edge of a cut the two sides disagree, and the answer is None -- the
link stays on X, which also lands on the second, just in another tab. A
wrong minute on YouTube would be worse than the right one on X; checked
on 216 probes, the unbracketed rule sent 4 of them somewhere that did not
match.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PATH = Path(__file__).resolve().parent.parent / "data" / "youtube_map.json"

# Two anchors bracket a second only if they are this close together: one
# anchor lands about every ten seconds of matched speech, so a wider gap
# means speech between them did not match, which is what a cut looks like.
_MAX_GAP = 30.0
# And only if their offsets agree this closely. The copies are the same
# recording, so within a stretch the offset does not drift.
_SAME_OFFSET = 4.0


@lru_cache(maxsize=1)
def _anchors() -> dict[str, list[list]]:
    try:
        return json.loads(_PATH.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("could not read youtube_map.json")
        return {}


def youtube_moment(episode_id: str, second: float) -> tuple[str, int] | None:
    """(youtube_video_id, second) for a moment in a live broadcast, or None."""
    rows = _anchors().get(episode_id)
    if not rows:
        return None
    keys = [r[0] for r in rows]
    i = bisect.bisect_right(keys, second)
    if i == 0 or i == len(rows):
        return None
    before, after = rows[i - 1], rows[i]
    if before[1] != after[1] or after[0] - before[0] > _MAX_GAP:
        return None
    offset_before, offset_after = before[2] - before[0], after[2] - after[0]
    if abs(offset_before - offset_after) > _SAME_OFFSET:
        return None
    return before[1], max(0, int(round(second + offset_before)))


# How far into a passage a citation can fall. A passage is about two and a
# half minutes of speech; the whole of it has to be on YouTube before the
# passage plays there, or a citation near its end lands in a cut.
_PASSAGE_REACH = 150.0


def youtube_first(hit: dict) -> dict:
    """A search passage, pointed at YouTube when the whole of it is there.

    Adds `youtube_offset`, the seconds to add to the broadcast's clock to
    get YouTube's. The answer quotes the broadcast's clock -- "around
    36:47" -- because that is the transcript it read, so the page adds the
    offset when it opens the player rather than the answer being rewritten.
    Anything that does not map whole comes back unchanged, on X.
    """
    episode = str(hit.get("episode_id") or "")
    if not episode.startswith("x-"):
        return hit
    start = float(hit.get("start_seconds") or 0)
    first = youtube_moment(episode, start)
    last = youtube_moment(episode, start + _PASSAGE_REACH)
    if not first or not last or first[0] != last[0]:
        return hit
    offset = first[1] - start
    if abs((last[1] - (start + _PASSAGE_REACH)) - offset) > _SAME_OFFSET:
        return hit
    return {**hit,
            "deep_link": f"https://www.youtube.com/watch?v={first[0]}&t={first[1]}s",
            "youtube_offset": round(offset),
            "live_link": hit.get("deep_link")}


def summary_moments(episode_id: str, summary: str) -> dict[str, list]:
    """For each [h:mm:ss] in a summary: [youtube_id, second], where it maps."""
    out: dict[str, list] = {}
    if not episode_id.startswith("x-"):
        return out
    for h, m, s in re.findall(r"\[(?:(\d+):)?(\d{1,2}):(\d{2})\]", summary or ""):
        second = int(h or 0) * 3600 + int(m) * 60 + int(s)
        found = youtube_moment(episode_id, second)
        if found:
            out[str(second)] = [found[0], found[1]]
    return out
