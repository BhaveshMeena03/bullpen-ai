"""How much of this show is actually indexed, once the same show stops
counting twice.

    .venv/bin/python scripts/count_the_archive.py
    .venv/bin/python scripts/count_the_archive.py --show-groups

Every public number about this project comes from here: the site, the
listing application, and anything posted about the size of the archive.
It exists because that number was wrong in public once already -- 92.7
hours, which was the sum of the files rather than the length of the show.

The duplicates are not copies. A broadcast goes out live on X and the cut
goes up on YouTube the next day, and the two are transcribed separately
from different audio, so the text is similar but never identical. That is
why the obvious tests all fail:

  file count        counts the same show up to four times
  word overlap      two hours of crypto talk share a vocabulary; distinct
                    episodes look like duplicates
  n-gram shingles   two transcriptions of one conversation share almost no
                    exact six-word runs; real duplicates look distinct

What does survive a second transcription is the rare words -- the guest's
name, a ticker, a number somebody said once. Whisper writes those
consistently and other episodes do not contain them. So a pair is the
same show when most of the smaller one's rare vocabulary appears in the
larger, and they aired within a few days of each other.

Pairs are merged transitively, since one broadcast can have four cuts,
and the longest member of each group is the one that survives -- the full
broadcast, not somebody's highlight of it.

The check that the grouping is right is not the threshold, it is the
answer: the groups come out as one show a week from 1 May to 3 September,
which is Market Bubble #1 through #18 in order, with only the week of
28 May missing -- a broadcast that has not been ingested yet. Nothing in
here knows the episode numbers. Reconstructing them is the evidence.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPISODES = ROOT / "data" / "episodes.json"

WORD = re.compile(r"[a-z][a-z']{3,}")
# A word in three files or fewer is specific to a conversation. Above that
# it is the vocabulary of the show and says nothing about which episode.
RARE_DF = 3
# Half the smaller one's rare vocabulary. Measured across every pair, and
# there is a clean gap to put the line in: every pair scoring 0.42 or above
# is genuinely one show, and no unrelated pair reaches 0.30. Nothing at all
# falls between. 0.40 sits in the empty band.
#
# It was 0.60 first, which read as safe and quietly split the 20 August
# show into three: the live broadcast at 5.4h, the Orangie cut at 1.0h and
# the #16 upload the next day, scoring 0.53, 0.57 and 0.43 against each
# other. A threshold above the data is not caution, it is a wrong answer
# that looks careful.
SAME_SHOW = 0.40
# A cut usually goes up the next day. "Inside Ansem's Trade Journal" went
# up four days after the show it came from, so the window is a week.
SAME_WEEK_DAYS = 6


def words(episode: dict) -> set[str]:
    return set(WORD.findall(
        " ".join(s.get("text", "") for s in episode["segments"]).lower()))


def seconds(episode: dict) -> float:
    return max((s.get("t", 0) for s in episode.get("segments") or []),
               default=0)


def aired(episode: dict) -> datetime.date | None:
    stamp = (episode.get("published_at") or episode.get("date") or "")[:10]
    try:
        return datetime.date.fromisoformat(stamp)
    except ValueError:
        return None


def group(episodes: list[dict]) -> list[list[dict]]:
    """Episodes clustered so each cluster is one show."""
    vocabulary = {e["episode_id"]: words(e) for e in episodes}
    seen: collections.Counter = collections.Counter()
    for bag in vocabulary.values():
        seen.update(bag)
    rare = {w for w, n in seen.items() if n <= RARE_DF}
    marks = {k: v & rare for k, v in vocabulary.items()}

    parent = {e["episode_id"]: e["episode_id"] for e in episodes}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(episodes):
        for b in episodes[i + 1:]:
            first, second = marks[a["episode_id"]], marks[b["episode_id"]]
            if not first or not second:
                continue
            when_a, when_b = aired(a), aired(b)
            if not when_a or not when_b:
                continue
            if abs((when_a - when_b).days) > SAME_WEEK_DAYS:
                continue
            smaller = first if len(first) <= len(second) else second
            larger = second if len(first) <= len(second) else first
            if len(smaller & larger) / len(smaller) > SAME_SHOW:
                ra, rb = root(a["episode_id"]), root(b["episode_id"])
                if ra != rb:
                    parent[ra] = rb

    clusters: dict[str, list[dict]] = collections.defaultdict(list)
    for e in episodes:
        clusters[root(e["episode_id"])].append(e)
    return list(clusters.values())


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
