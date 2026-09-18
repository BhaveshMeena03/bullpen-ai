"""Does each summary credit the host who was actually talking?

    .venv/bin/python scripts/audit_summary_speakers.py
    .venv/bin/python scripts/audit_summary_speakers.py --only x-2100684090018316301

The summaries are the one surface no guard reads. attribution.correct
checks quotes in answers; nothing checked a summary line like "Ansem
live-trades Dividend Hounds", which was Banks, and which went on the site
and into a drafted post before a human watching the show caught it.

For every summary line that names a host and carries a timestamp, this
reads the speaker labels over the stretch that line describes and sorts
it into one of three:

    contradicted  the named host does not hold the floor there and the
                  other host does. Almost certainly wrong.
    unsupported   no host is labelled there, so nothing can confirm it.
                  Not wrong, but a name the evidence does not back.
    ok            the named host is labelled speaking in that stretch.

The stretch runs a minute before the stamp to five after it, because a
topic bullet's timestamp is where the topic begins and the thing it
credits is usually inside it rather than at it. Loose on purpose: a
check that cries wolf gets ignored, and the point is the contradicted
set, which a loose window makes smaller rather than larger.

Exits non-zero when anything is contradicted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import episode_store  # noqa: E402
from app.citations import readings  # noqa: E402
from app.summaries import SummaryStore  # noqa: E402

SPEAKER_MAP = ROOT / "data" / "speaker_map.json"

BEFORE, AFTER = 60, 180
# The floor, in labelled lines, for a host to count as speaking in a stretch.
HOLDS = 3
# And the share of the stretch's labelled speech the credited host must
# have. A count alone passed the one line this was written to catch: over
# ep 20's trade Ansem has 5 labelled lines to Banks's 39, clearing any
# small floor while being plainly not his segment -- he is teasing the
# caller while Banks trades. Share is what separates holding the floor
# from chiming in.
SHARE = 0.25

_HOST = {"ansem": "Ansem", "banks": "FaZe Banks"}
# A host CREDITED with something: the name, then (after at most one word,
# "Ansem also notes") a verb of saying or doing. A bare mention is not a
# credit. The first run flagged "celebrating Ansem's 1 million follower
# milestone" and "comparing Ansem to Donald Trump" as misattributed --
# Ansem is the subject there, not the speaker -- which is the distinction
# the summarizer's own prompt draws and this has to draw too.
_VERB = (r"(?:says?|said|notes?|noted|argues?|argued|explains?|explained|"
         r"reveals?|revealed|recounts?|describes?|described|admits?|jokes?|"
         r"asks?|asked|calls?|called|claims?|pushes|pushed|predicts?|"
         r"reflects?|shares?|shared|mentions?|mentioned|thinks|believes|"
         r"warns?|trades?|traded|live-trades?|buys?|bought|sells?|sold|"
         r"bets?|walks|breaks|makes|made|gives|gave|pitches|tells|told|"
         r"responds?|counters?|questions?|pulls|hands|announces?|teases?)")
_NAMES = re.compile(rf"(?i)\b(ansem|banks)\b(?!'s)(?:\s+\w+)?\s+{_VERB}\b")
_STAMP = re.compile(r"\[?(\d{1,2}:\d{2}(?::\d{2})?)")


def label_counts(episode: dict, speakers: dict, start: int, end: int) -> dict:
    """How many labelled lines each host has between two seconds."""
    spoken = [s for s in episode["segments"] if (s.get("text") or "").strip()]
    counts = {"Ansem": 0, "FaZe Banks": 0}
    for index, segment in enumerate(spoken):
        if start <= segment.get("t", 0) <= end:
            who = speakers.get(str(index))
            if who in counts:
                counts[who] += 1
    return counts


def verdicts(summary: str, episode: dict, speakers: dict):
    """(verdict, line, detail) for every host-naming, stamped line."""
    for line in summary.splitlines():
        named = {_HOST[n.lower()] for n in _NAMES.findall(line)}
        stamp = _STAMP.search(line)
        if not named or not stamp:
            continue
        at = readings(stamp.group(1))[0]
        counts = label_counts(episode, speakers, at - BEFORE, at + AFTER)
        for host in sorted(named):
            other = next(h for h in counts if h != host)
            total = counts[host] + counts[other]
            share = counts[host] / total if total else 0.0
            if counts[host] >= HOLDS and share >= SHARE:
                verdict = "ok"
            elif counts[other] >= HOLDS:
                verdict = "contradicted"
            else:
                verdict = "unsupported"
            yield verdict, line.strip(), f"{host}: {counts[host]} lines, {other}: {counts[other]}"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[])
    args = ap.parse_args()

    speaker_map = json.loads(SPEAKER_MAP.read_text())
    episodes = {e["episode_id"]: e for e in episode_store.load()}
    rows = await SummaryStore().list_all()
    tally = {"ok": 0, "contradicted": 0, "unsupported": 0}

    for row in rows:
        episode_id = row.get("episode_id")
        if args.only and episode_id not in args.only:
            continue
        episode = episodes.get(episode_id)
        if not episode:
            continue
        found = list(verdicts(row.get("summary", ""), episode,
                              speaker_map.get(episode_id, {})))
        if not found:
            continue
        print(f"\n  {row.get('published_at', '')[:10]}  {episode['title'][:60]}")
        for verdict, line, detail in found:
            tally[verdict] += 1
            if verdict != "ok":
                print(f"    {verdict.upper():12} {line[:220]}")
                print(f"    {'':12} {detail}")

    print(f"\n  {tally['ok']} ok · {tally['contradicted']} contradicted · "
          f"{tally['unsupported']} unsupported\n")
    return 1 if tally["contradicted"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
