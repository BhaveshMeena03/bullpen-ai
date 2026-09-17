"""Did the answer keep the figure the transcript actually said?

    .venv/bin/python scripts/verify_numbers.py
    .venv/bin/python scripts/verify_numbers.py --limit 6

The fourth thing nothing checked. The others ask whether the right
passage was found, whether the right person was credited, whether the
quote sits near the cited second, and whether the right asset was
named. None of them looks at the numbers, and this went out:

    transcript:  "your buy targets for bitcoin is like 55k,
                  solana's 55 and like hyperliquid was like 55"
    answer:      "his buy targets were $58K for Bitcoin, $58 for S..."

55 became 58. Right episode, right second, right speaker, right asset,
wrong number -- and a wrong number is the one error a reader can neither
detect nor forgive on a tool whose whole claim is precision.

Checked against the lines AROUND each cited moment rather than the whole
bag of passages, for the same reason verify_entities is: a figure
mentioned two minutes later in a different sentence vouches for nothing.

Exits non-zero on any figure the passages do not carry.
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

_LINE = re.compile(r"^\[(\d[\d:]*)\]\s*(.*)$", re.M)
_STAMP = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")

# A figure worth checking: at least two digits, so "one of the hosts" and
# "3 things" do not fill the report with noise. Percentages and money
# keep their shape; the suffix is normalised away below.
_FIGURE = re.compile(r"\$?\b(\d[\d,]*\.?\d*)\s*(k|m|b|bn|%|million|billion|thousand)?\b",
                     re.I)

NEAR_SECONDS = 20

# When the show aired, in the shapes an answer writes it. Not a claim
# about what anybody said, so the figures inside are not checkable
# against the transcript.
_DATE = re.compile(
    r"""(?ix)
    \b(?: jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec )
      [a-z]* \s+ \d{1,2} (?: \s*,\s* \d{4} )?     # May 14, 2026
    | \b\d{1,2} \s+
      (?: jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec )[a-z]*
    | \b(?:19|20)\d{2}\b                          # a bare year
    | \bep(?:isode)?\.?\s*\#?\s*\d{1,3}\b         # episode 19
    """)

_SCALE = {"k": "000", "thousand": "000", "m": "000000", "million": "000000",
          "b": "000000000", "bn": "000000000", "billion": "000000000"}


def _seconds(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def figures(text: str) -> set[str]:
    """Every figure in the text, as the bare digits a reader would hear.

    "55k", "55,000" and "55 thousand" all reduce to "55", because the
    transcript and the answer rarely agree on the notation and the thing
    being checked is the number itself. The scale is kept alongside so a
    claim of millions is not vouched for by a figure in thousands.
    """
    out: set[str] = set()
    for value, suffix in _FIGURE.findall(text or ""):
        digits = value.replace(",", "").rstrip(".")
        if not digits or len(digits.lstrip("0").replace(".", "")) < 2:
            continue
        out.add(digits)
        if not suffix:
            continue
        # Multiplied, not concatenated. "1.2 million" became the string
        # "1.2000000" on the first run, which matches nothing and
        # reported a correct answer as unsupported.
        zeros = _SCALE.get(suffix.lower())
        if not zeros:
            continue
        try:
            scaled = float(digits) * (10 ** len(zeros))
        except ValueError:
            continue
        out.add(str(int(scaled)) if scaled.is_integer() else str(scaled))
    return out


def _lines_near(hits, want: int) -> str:
    out = []
    for hit in hits or ():
        for stamp, said in _LINE.findall(getattr(hit, "text_ts", "") or ""):
            try:
                at = _seconds(stamp)
            except ValueError:
                continue
            if abs(at - want) <= NEAR_SECONDS:
                out.append(said)
    return " ".join(out)


def unsupported(answer: str, hits) -> list[str]:
    """Figures the answer states that the cited moment does not carry."""
    stamps = _STAMP.findall(answer or "")
    if not stamps:
        return []
    here = " ".join(_lines_near(hits, _seconds(s)) for s in stamps[:4])
    if not here.strip():
        return []
    have = figures(here)
    if not have:
        return []

    # Only figures in a sentence that cites a moment. An answer covering
    # three moments states figures from all of them and cites one, so
    # checking every number against one window reported four correct
    # answers as wrong on the first run. A figure sitting beside a
    # timestamp is a claim about THAT moment and is checkable; one in a
    # summarising sentence is not.
    missing: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", answer or ""):
        if not _STAMP.search(sentence):
            continue
        # The timestamps themselves are not claims about the world, and
        # neither is the date the episode aired. "Around 5:29 in the May
        # 14 episode" reported 14 as an unsupported figure in three
        # different questions before this: the answer naming when a show
        # went out is not the answer stating a number somebody said.
        clean = _STAMP.sub(" ", sentence)
        clean = _DATE.sub(" ", clean)
        missing |= figures(clean) - have
    return sorted(missing)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pause", type=float, default=2.0)
    args = ap.parse_args()

    questions = [
        "what did ansem say about solana price targets",
        "what did they say about tom lee",
        "how much revenue is pump fun making",
        "what did they say about bonk market cap",
        "what price did they say bitcoin would hit",
        "how much did the show say someone turned 500 dollars into",
        "what percentage did they say about the fee split",
        "what did they say about hyperliquid revenue",
        "how many users did they say fomo has",
        "what did they say about mstr premium",
    ]
    asked = questions[:args.limit] if args.limit else questions
    index = PodcastIndex()
    bad: list[str] = []

    for n, question in enumerate(asked, 1):
        try:
            result = await index.search(question)
        except Exception as exc:                                # noqa: BLE001
            print(f"  {n:2}. ERROR  {question[:46]} ({type(exc).__name__})")
            continue
        missing = unsupported(result.answer or "", result.hits)
        print(f"  {n:2}. {'FAIL ' if missing else 'ok   '}  {question[:52]}")
        for figure in missing:
            line = f"{question[:32]}: says {figure}, the passages do not"
            print(f"        {line}")
            print(f"        answer: "
                  f"{' '.join((result.answer or '').split())[:120]}")
            bad.append(line)
        await asyncio.sleep(args.pause)

    print(f"\n  {len(asked)} asked · {len(bad)} figure(s) the passages "
          f"do not support")
    if bad:
        print("\n  FAILURES:")
        for line in bad:
            print(f"     {line}")
    print()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
