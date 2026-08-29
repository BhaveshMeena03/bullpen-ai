"""Answers that deny and then answer anyway.

    .venv/bin/python scripts/check_hedging.py

"I don't see a direct statement from Ansem about paying attention to
crypto before everyone else. However, around 1:46:25 he does discuss..."

Both halves of that go out as a public reply, and the reader stops at the
first sentence. The answer is right; the opening tells them it is not.

The shape is mechanical, so it can be counted rather than eyeballed: the
answer carries a citation AND opens with a denial. An answer that only
denies is fine -- that is a real refusal, and the twenty questions in the
ABSENT set must keep producing them. An answer that only cites is fine.
It is the pair that is the defect.

Run before and after a prompt change. Eyeballing one example is how the
first attempt at this shipped while still leaking.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.podcast import PodcastIndex  # noqa: E402
from scripts.ab_questions import ABSENT  # noqa: E402

# A citation: "around 1:46:25", "at 27:09". The model is told to write
# them in prose, so this is the shape they actually take.
CITES = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

# Denials, as they appear at the START of an answer.
DENIES = re.compile(
    r"^\W*(?:"
    r"i (?:couldn't|could not|can't|cannot|don't|do not)\b"
    r"|i (?:didn't|did not) (?:find|see)\b"
    r"|there(?:'s| is) (?:no|nothing|not)\b"
    r"|no (?:direct|specific|clear) (?:statement|mention|quote|passage)\b"
    r"|not (?:in|that|word for word|exactly)\b"
    r"|nothing (?:in|matching|specific)\b"
    r"|the (?:excerpts|transcripts) (?:don't|do not)\b"
    r")", re.I)

# Questions whose match is loose enough to tempt a hedge. Every one was
# observed hedging, or is close in shape to one that did.
LOOSE = [
    "what did ansem say about paying attention to crypto before everyone else",
    "what did ansem say about being early innings",
    "Ansem saying, like, we need to onboard girls.",
    "what did ansem say about making a plan before you buy",
    "what did banks say about gambling",
    "what did ansem say about doing research before consensus",
    "what did they say about women in crypto",
    "what did ansem say about the four year cycle",
    "what did banks say about his own trading",
    "what did ansem say about taking profits",
    "who said trading is becoming the new celebrity culture",
    "what did they say about buying the dip",
    "what did ansem say about conviction",
    "what did banks say about being blindsided",
    "what did they say about q4 lows",
]


def hedged(answer: str) -> bool:
    """Denies in the opening, then cites something anyway."""
    first = answer.strip().split("\n")[0]
    return bool(DENIES.search(first)) and bool(CITES.search(answer))


async def ask(index: PodcastIndex, q: str, tries: int = 3):
    for attempt in range(tries):
        try:
            return await index.search(q)
        except Exception:                                       # noqa: BLE001
            if attempt + 1 == tries:
                raise
            await asyncio.sleep(6)
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=1.5)
    args = ap.parse_args()

    index = PodcastIndex()
    hedges = clean = 0
    print(f"\n  {len(LOOSE)} loose-match questions\n")
    for n, q in enumerate(LOOSE, 1):
        r = await ask(index, q)
        if hedged(r.answer):
            hedges += 1
            print(f"  {n:2}. HEDGED  {q[:52]}")
            print(f"        {r.answer.strip().splitlines()[0][:104]}")
        else:
            clean += 1
        await asyncio.sleep(args.pause)
    print(f"\n  hedged {hedges}/{len(LOOSE)} · clean {clean}")

    # A real refusal must survive the fix. Nothing here is in any
    # transcript, so denying is the only correct answer.
    print(f"\n  {len(ABSENT)} absent topics — every one should still refuse\n")
    answered = 0
    for n, q in enumerate(ABSENT, 1):
        r = await ask(index, q)
        first = r.answer.strip().split("\n")[0]
        if not DENIES.search(first):
            answered += 1
            print(f"  {n:2}. NO LONGER REFUSES  {q[:46]}")
            print(f"        {r.answer[:110]}")
        await asyncio.sleep(args.pause)
    print(f"\n  still refusing {len(ABSENT) - answered}/{len(ABSENT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
