"""What people have asked, and what it could not answer.

    .venv/bin/python scripts/questions.py
    .venv/bin/python scripts/questions.py --misses
    .venv/bin/python scripts/questions.py --days 7

The misses are the point. An answered question confirms something already
known; a miss names an episode worth indexing, a person whose name
retrieval does not recognise, or a thing people expect this to do that it
does not. None of that can be guessed from the code.

Reads only. Costs nothing but a Pinecone list.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.questions import QuestionLog  # noqa: E402

# Words that say nothing about the subject. Without this the top of the
# table is "what", "did", "the" every time.
DULL = {
    "what", "did", "does", "the", "and", "for", "was", "were", "say", "said",
    "about", "who", "when", "how", "why", "with", "this", "that", "they",
    "his", "her", "their", "you", "your", "any", "has", "have", "had", "are",
    "from", "there", "them", "then", "than", "which", "would", "could",
    "mbubblesearch", "episode", "show", "podcast", "tell",
}


def topics(rows: list[dict], n: int = 12) -> list[tuple[str, int]]:
    words: Counter[str] = Counter()
    for row in rows:
        for word in re.findall(r"[a-z0-9']{3,}", row.get("question", "").lower()):
            if word not in DULL:
                words[word] += 1
    return words.most_common(n)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", action="store_true",
                    help="only the questions it could not answer")
    ap.add_argument("--days", type=int, help="only the last N days")
    ap.add_argument("--replies", action="store_true",
                    help="show what the bot actually replied, which is the "
                         "only way to spot a confident wrong answer")
    ap.add_argument("--limit", type=int, default=40,
                    help="how many questions to print")
    args = ap.parse_args()

    rows = await QuestionLog().list_all()
    if not rows:
        print("\n  Nothing logged yet. The log starts filling from the next "
              "reply the bot posts.\n")
        return 0

    if args.days:
        cutoff = (datetime.now(UTC) - timedelta(days=args.days)).isoformat()
        rows = [r for r in rows if r.get("asked_at", "") >= cutoff]

    missed = [r for r in rows if not r.get("answered", True)]
    askers = Counter(r.get("asker") or "unknown" for r in rows)

    print(f"\n  {len(rows)} question(s) · {len(missed)} unanswered "
          f"({100 * len(missed) / max(len(rows), 1):.0f}%)")
    print(f"  {len(askers)} distinct asker(s)")

    # Worth seeing plainly: a tool used by one person is being tested, not
    # used, and the numbers below mean something different in that case.
    top, count = askers.most_common(1)[0]
    if count > len(rows) * 0.6 and len(rows) > 3:
        print(f"  {100 * count / len(rows):.0f}% of these are @{top} — "
              f"still mostly self-testing")

    shown = missed if args.misses else rows
    label = "COULD NOT ANSWER" if args.misses else "RECENT"
    print(f"\n  ── {label} ──\n")
    for row in shown[:args.limit]:
        mark = " " if row.get("answered", True) else "!"
        when = (row.get("asked_at") or "")[:16].replace("T", " ")
        who = row.get("asker") or "?"
        print(f"  {mark} {when}  {str(who)[:18]:18}  "
              f"{row.get('question','')[:78]}")
        if args.replies and row.get("reply"):
            body = " ".join(row["reply"].split())
            print(f"          -> {body[:140]}")

    if not args.misses and missed:
        print(f"\n  ── {len(missed)} of these were misses. "
              f"--misses to see only those ──")

    if len(rows) > 3:
        print("\n  ── most asked about ──")
        for word, n in topics(rows):
            print(f"     {n:3}  {word}")
        if missed:
            print("\n  ── most asked about, among the misses ──")
            print("     (each of these is an episode to index or a name "
                  "retrieval does not know)")
            for word, n in topics(missed, 8):
                print(f"     {n:3}  {word}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
