"""How much of this show is actually indexed, once the same show stops
counting twice.

    .venv/bin/python scripts/count_the_archive.py
    .venv/bin/python scripts/count_the_archive.py --show-groups

Every public number about this project comes from here: the site, the
listing application, and anything posted about the size of the archive.
It exists because that number was wrong in public once already -- 92.7
hours, which was the sum of the files rather than the length of the show.

The grouping itself lives in app/dedupe.py, because the website needs the
same answer and two implementations of one rule is how they come to
disagree. They did: this script and the site once reported 18 and 20 for
the same archive.

The check that the grouping is right is not the threshold, it is the
answer: the groups come out as one show a week from 1 May to 3 September,
which is Market Bubble #1 through #18 in order, with only the week of
28 May missing -- a broadcast that has not been ingested yet. Nothing in
the grouping knows the episode numbers. Reconstructing them is the
evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.dedupe import group_by_show, _length as seconds   # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"


def group(episodes: list[dict]) -> list[list[dict]]:
    """One cluster per broadcast. The rule is app/dedupe.py's."""
    return group_by_show(episodes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-groups", action="store_true")
    args = ap.parse_args()

    episodes = json.loads(EPISODES.read_text())
    clusters = group(episodes)
    kept = [max(c, key=seconds) for c in clusters]

    raw_hours = sum(seconds(e) for e in episodes) / 3600
    hours = sum(seconds(e) for e in kept) / 3600
    lines = sum(len(e.get("segments") or []) for e in kept)

    print(f"\n  {len(episodes)} files in the index")
    print(f"  {len(clusters)} distinct shows")
    print(f"  {hours:.1f} hours   (the files add up to {raw_hours:.1f})")
    print(f"  {lines:,} transcript lines\n")
    print(f"  Say: {len(clusters)} shows, {hours:.0f} hours.\n")

    if args.show_groups:
        for cluster in sorted(clusters, key=len, reverse=True):
            if len(cluster) == 1:
                continue
            for i, e in enumerate(sorted(cluster, key=seconds, reverse=True)):
                mark = "keep" if i == 0 else "  ->"
                print(f"  {mark} {seconds(e)/3600:4.1f}h  "
                      f"{(e.get('published_at') or '')[:10]}  {e['title'][:46]}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
