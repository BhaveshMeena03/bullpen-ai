"""Did the answer name the right asset, and the right person?

    .venv/bin/python scripts/verify_entities.py
    .venv/bin/python scripts/verify_entities.py --limit 6

The three checkers already here ask different questions:

    hundred_questions    did it find the right passage
    verify_attribution   did it name the right speaker
    verify_replies       is the quote near the cited second

None of them asks whether the answer named the right THING, and that is
how this went out in public:

    asked what the show said about Tom Lee, the reply said he "bought
    roughly 15 billion of Solana"

He bought ETH. Retrieval was perfect -- right episode, right second,
3:09:37, the only Tom Lee mention in the archive. What failed is one
level down. Whisper wrote ETH as a bare "e":

    [3:08:53] soul e chart is the one bro
    [3:09:45] bought it as and just ridiculous amount of e

"soul e" is SOL/ETH. The model read an ambiguous letter in a
Solana-heavy episode and resolved it to Solana, which also inverts the
argument being made: Tom Lee buying ETH is WHY that ratio topped out.

So this checks entities rather than passages. For every asset the answer
names, the passages the model was shown have to contain that asset, by
name or by an unambiguous ticker. A bare "e" vouches for nothing, which
is the point -- it cannot support Ethereum either, and an answer resting
on it is guessing whichever way it lands.

Exits non-zero on any unsupported entity.
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

# asset -> the spellings that genuinely vouch for it in a transcript.
# Deliberately excludes single letters and bare symbols: "e" and "sol"
# on their own are what this exists to catch. Whisper writes tickers as
# words far more often than as symbols, so the words are what to look
# for.
ASSETS: dict[str, tuple[str, ...]] = {
    "solana": ("solana", "sol ", "$sol"),
    "ethereum": ("ethereum", "eth ", "$eth", "ether"),
    "bitcoin": ("bitcoin", "btc", "$btc"),
    "hyperliquid": ("hyperliquid", "hype"),
    "zcash": ("zcash", "zec"),
    "bonk": ("bonk",),
    "pump": ("pump",),
    "chainlink": ("chainlink", "link"),
    "dogecoin": ("dogecoin", "doge"),
    "xrp": ("xrp", "ripple"),
}

# People are NOT checked this way, and the attempt is worth recording.
#
# An answer names people for two different reasons, and only one of them
# is a claim: "Brian Armstrong said X" is checkable, but "in the episode
# with Erik Voorhees and Mike Majlak" is the show's billing and says
# nothing about who spoke. Checking names flagged Voorhees four times in
# fifteen questions on exactly that, and Armstrong on an answer that was
# correct -- his name is never SPOKEN in his own interview, it comes off
# the guest labels.
#
# That is the same false positive that broke verify_attribution this
# morning, arrived at from the other direction. Speakers are already
# checked properly there, against labelled lines rather than raw text.
# This file checks what the words REFER to, which is the gap nothing
# else covered.
PEOPLE: dict[str, tuple[str, ...]] = {}

QUESTIONS = [
    # The one that failed in public.
    "what did they say about tom lee",
    "what did they say about tom lee buying",
    # Assets that are easy to confuse with each other.
    "what did ansem say about the sol eth ratio",
    "what did they say about ethereum treasury companies",
    "what did ansem say about solana price targets",
    "what did they say about zcash",
    "what did ansem say about hyperliquid",
    "what did they say about bitcoin dominance",
    "what did ansem say about bonk",
    "what did they say about pump fun revenue",
    # People, where the wrong name is a claim about a real person.
    "what did they say about vlad tenev",
    "what did brian armstrong say about coinbase",
    "what did they say about michael saylor",
    "what did they say about trump and crypto",
    "what did they say about elon and doge",
]


_LINE = re.compile(r"^\[(\d[\d:]*)\]\s*(.*)$", re.M)
_STAMP = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")

# How far either side of the cited second still counts as "here".
#
# Tight on purpose. The Tom Lee passage discusses SOL and ETH in the same
# breath: he is named at 3:09:37, "amount of e" lands at 3:09:45, and
# "if you wanted to go long solana" at 3:10:24. A sixty-second window
# swallows that last line and the check passes on a word spoken about a
# different sentence. Fifteen seconds keeps the claim and its object
# together, which is the whole point.
NEAR_SECONDS = 15


def _seconds(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _lines_near(hits, want: int) -> str:
    """Transcript lines within NEAR_SECONDS of a cited moment."""
    out = []
    for hit in hits or ():
        for stamp, said in _LINE.findall(getattr(hit, "text_ts", "") or ""):
            try:
                at = _seconds(stamp)
            except ValueError:
                continue
            if abs(at - want) <= NEAR_SECONDS:
                out.append(said)
    return " ".join(out).lower()


def unsupported(answer: str, hits) -> list[str]:
    """Entities the answer names that the cited moment does not mention.

    Checked against the lines AROUND the cited second, not the whole bag
    of retrieved passages. The bag is too coarse to catch anything: the
    Tom Lee passage mentions Solana forty-seven seconds later, in a
    different sentence about a different thing, and that was enough to
    vouch for "Tom Lee bought Solana" when the line itself says "e".
    """
    said = (answer or "").lower()
    stamps = _STAMP.findall(answer or "")
    if not stamps:
        return []
    here = " ".join(_lines_near(hits, _seconds(s)) for s in stamps[:4])
    if not here.strip():
        return []                      # unstamped passages: nothing to check
    words = set(re.findall(r"[a-z0-9$']+", here))
    problems = []
    for label, spellings in (*ASSETS.items(), *PEOPLE.items()):
        if not re.search(rf"\b{re.escape(label)}\b", said):
            continue
        if any(s in here for s in spellings):
            continue
        # Whisper misspells names constantly -- "salana" for Solana,
        # "Anthem" for Ansem, "Stomp" for Stonk. Exact matching reported
        # a correct answer about Bonk as unsupported because the line
        # said "onchain coins on salana doing well". A near-miss on a
        # word of this length is the same word.
        if any(_near_miss(label, w) for w in words):
            continue
        problems.append(label)
    return problems


def _near_miss(want: str, got: str) -> bool:
    """Same word, one keystroke out. Whisper's spelling, not a new entity."""
    if len(want) < 5 or abs(len(want) - len(got)) > 1:
        return False
    if want == got:
        return True
    # One substitution, or one insertion/deletion. Enough for salana ->
    # solana without letting "bonk" match "bank".
    if len(want) == len(got):
        # strict=True: equal lengths are already guaranteed above, so a
        # mismatch here would mean the guard is wrong rather than the input.
        return sum(a != b for a, b in zip(want, got, strict=True)) == 1
    short, long_ = (want, got) if len(want) < len(got) else (got, want)
    # Both forms have to be substantial. A one-letter deletion turns
    # "zcash" into "cash", which is a different word, not a misspelling.
    if len(short) < 5:
        return False
    i = 0
    for ch in long_:
        if i < len(short) and short[i] == ch:
            i += 1
    return i == len(short)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pause", type=float, default=2.0)
    args = ap.parse_args()

    index = PodcastIndex()
    asked = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    bad: list[str] = []

    for n, question in enumerate(asked, 1):
        try:
            result = await index.search(question)
        except Exception as exc:                                # noqa: BLE001
            print(f"  {n:2}. ERROR  {question[:46]} ({type(exc).__name__})")
            continue
        problems = unsupported(result.answer or "", result.hits)
        print(f"  {n:2}. {'FAIL ' if problems else 'ok   '}  {question[:52]}")
        for label in problems:
            line = (f"{question[:34]}: named {label!r}, no passage mentions it")
            print(f"        {line}")
            print(f"        answer: "
                  f"{' '.join((result.answer or '').split())[:110]}")
            bad.append(line)
        await asyncio.sleep(args.pause)

    print(f"\n  {len(asked)} asked · {len(bad)} entity claim(s) the "
          f"passages do not support")
    if bad:
        print("\n  FAILURES:")
        for line in bad:
            print(f"     {line}")
    print()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
