"""How far apart are two recordings of the same show?

    .venv/bin/python scripts/align_recordings.py 5mXHC5Hu9Wc x-2052486417154969659
    .venv/bin/python scripts/align_recordings.py --at 56:30 5mXHC5Hu9Wc x-205248...

Most shows are in the archive twice: the YouTube upload and the X
broadcast it was cut from. They are the same conversation with different
clocks -- the stream carries a waiting screen and a pre-show the upload
drops, so the same sentence sits about three minutes later on X.

That matters because naming voices needs a human saying who was on air
between which times, and those times are read off whichever recording the
person watched. Knowing the offset makes one answer cover both copies:
ep 2 turned two windows into 1,623 labelled lines instead of 645.

Matching whole transcript lines does not work. The two copies were
transcribed separately, so Whisper split and spelled them differently and
only two lines in the entire episode matched exactly. Eight-word runs
match instead -- 27,793 of them on that pair, and 27,776 agreed on the
offset within fifteen seconds, which is the kind of margin that needs no
argument.

Prints nothing but the offset and how much of the evidence agrees with
it. A pair that does not agree is not the same show, and saying so is the
point: applying a window across an alignment nobody checked is how a
guest's name lands on somebody else's half hour.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPISODES = ROOT / "data" / "episodes.json"
GRAM = 8
# Two runs of eight words can legitimately land a few seconds apart -- the
# segment carrying them starts where Whisper decided it started, and the
# two passes decided differently. Fifteen seconds is looser than any real
# disagreement and far tighter than a mismatched pair, which scatters.
AGREE_SECONDS = 15.0
# Below this, the pair is not two copies of one show and no offset should
# be trusted from it.
MIN_MATCHES = 200
MIN_AGREEMENT = 0.90


def _words(episode: dict) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for seg in episode["segments"]:
        for word in re.findall(r"[a-z0-9']+", seg.get("text", "").lower()):
            out.append((word, float(seg.get("t", 0.0))))
    return out


def _grams(words: list[tuple[str, float]]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = collections.defaultdict(list)
    for i in range(max(0, len(words) - GRAM)):
        key = " ".join(w for w, _ in words[i:i + GRAM])
        found[key].append(words[i][1])
    return found


def offset(first: dict, second: dict) -> tuple[float | None, int, float]:
    """(seconds to add to `first` to reach `second`, matches, agreement).

    Only runs that appear ONCE in each recording are counted. A phrase
    somebody repeats through the night would otherwise pair its first
    instance in one copy with its third in the other and invent a gap.
    """
    a, b = _grams(_words(first)), _grams(_words(second))
    deltas = [b[k][0] - ta[0] for k, ta in a.items()
              if len(ta) == 1 and len(b.get(k, ())) == 1]
    if len(deltas) < MIN_MATCHES:
        return None, len(deltas), 0.0
    middle = statistics.median(deltas)
    agree = sum(1 for d in deltas if abs(d - middle) < AGREE_SECONDS)
    return middle, len(deltas), agree / len(deltas)


def _seconds(stamp: str) -> float:
    parts = [float(p) for p in str(stamp).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--at", action="append", default=[],
                    help="a time in the first recording, translated for you")
    args = ap.parse_args()

    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}
    for eid in (args.first, args.second):
        if eid not in episodes:
            raise SystemExit(f"  {eid} is not in {EPISODES.name}")

    shift, matches, agreement = offset(episodes[args.first],
                                       episodes[args.second])
    if shift is None or agreement < MIN_AGREEMENT:
        print(f"\n  no alignment: {matches} matching runs, "
              f"{agreement * 100:.0f}% agreeing")
        print("  these are not two recordings of the same show — do not "
              "carry a window across them\n")
        return 1

    print(f"\n  {matches:,} matching runs, {agreement * 100:.2f}% agree")
    print(f"  {args.second} = {args.first} + {shift:.0f}s\n")
    for at in args.at:
        here = _seconds(at)
        print(f"    {_hms(here)}  ->  {_hms(here + shift)}")
    if args.at:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
