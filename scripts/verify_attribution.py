"""Check every name an answer uses against the passage it came from.

    .venv/bin/python scripts/verify_attribution.py
    .venv/bin/python scripts/verify_attribution.py --limit 10

The speaker labels let the model say "Ansem said" instead of "a guest
said". That is only an improvement if the name is right, and the first
attempt got it wrong in a way worth guarding against forever: asked what
Banks said about Solana, the retrieval filter handed over passages
containing both hosts, and a line prefixed "Ansem:" was reported as Banks
saying it — because the question had named Banks.

So this does not ask whether the answer looks good. For every quoted
fragment it finds the line in the passages the model was given, reads the
name prefixed to that line, and compares it to the name the answer used.
A mismatch is a false claim about a real person and is reported as a
failure, not a warning.

Paced deliberately. Fifty back-to-back searches time out against the
index at its configured read timeout, which is a fact about burst load
rather than a fault, so this waits between questions and retries once.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from app.podcast import PodcastIndex  # noqa: E402

SPEAKER_MAP = ROOT / "data" / "speaker_map.json"
GUEST_LABELS = ROOT / "data" / "guest_labels.json"

HOSTS = ("Ansem", "FaZe Banks", "Banks")

# Every name the archive can attribute a line to, not just the two hosts.
# Hard-coding the hosts made this blind to exactly the thing the guest
# labels add: an answer saying "Mizkif said X" matched nothing here and
# was skipped in silence, so a wrong guest attribution could never be
# reported. Built from the label files so it widens as they do, and
# falls back to the hosts alone when guest labels have not been written.
def _known_names() -> list[str]:
    names: set[str] = set(HOSTS)
    for path in (SPEAKER_MAP, GUEST_LABELS):
        if not path.exists():
            continue
        for rows in json.loads(path.read_text()).values():
            names.update(rows.values())
    # Longest first: "FaZe Banks" has to win against "Banks", or the
    # alternation matches the short one and the comparison below reads
    # the wrong claimed speaker.
    return sorted(names, key=len, reverse=True)


# "X said", "X noted", "X explained" — the shapes an attribution takes.
ATTRIBUTION = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in _known_names()) + r")\b"
    r"[^.!?\n]{0,40}?"
    r"\b(said|says|noted|explained|argued|described|recalled|admitted"
    r"|mentioned|claimed|stated|revealed|put it)\b")
QUOTED = re.compile(r'"([^"]{18,140})"')
LABELLED = re.compile(r"^\[[\d:]+\]\s*([A-Z][A-Za-z ]{2,20}):\s*(.+)$", re.M)

QUESTIONS = [
    "what did ansem say about hyperliquid",
    "what did ansem say about ethereum",
    "what did ansem say about pump fun",
    "what did ansem say about bonk",
    "what did ansem say about his solana price target",
    "what did banks say about streaming",
    "what did banks say about solana",
    "what did banks say about his portfolio",
    "what did banks say about content",
    "what did banks say about hyperliquid",
    "what did ansem say about airdrops",
    "what did banks say about faze",
    "what did ansem say about memecoins",
    "what did banks say about kick",
    "what did ansem say about trading",
    # Guest questions, weighted to the guests write_guest_labels.py can
    # actually claim. Without these the harness only ever asked about the
    # two hosts, so the guest labels -- the whole point of the change it
    # is meant to measure -- could neither pass nor fail here.
    "what did erik voorhees say about venice",
    "what did erik voorhees say about the government",
    "what did andrej say about grass",
    "what did andrej say about training data",
    "what did chris gilbert say about inference",
    "what did chris gilbert say about compute",
    "what did gpt live say about attention",
    "what did brez say about solana",
    "what did simple farmer say about robinhood",
    "what did lucas bruder say about solana",
    "what did cirrus say about nfts",
    "what did tjr say about tiktok",
    "what did jesse pollak say about coinbase",
    "what did sal qadir say about bullpen",
    "what did will clemente say about bitcoin",
    "what did al dunlap say about treasury",
    # Both hosts in one question. This is the shape that produced the
    # original bug -- passages containing both, and the model crediting
    # whichever name the question mentioned rather than the one on the
    # line -- and the shape the hyperliquid failure took again today.
    "what did ansem and banks say about hyperliquid",
    "who said what about solana between ansem and banks",
]


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']{4,}", text.lower())}


async def search(index: PodcastIndex, question: str, tries: int = 2):
    for attempt in range(tries):
        try:
            return await index.search(question)
        except TimeoutError:
            if attempt + 1 == tries:
                raise
            await asyncio.sleep(5)
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pause", type=float, default=2.0)
    args = ap.parse_args()

    index = PodcastIndex()
    asked = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    wrong: list[str] = []
    checked = unverifiable = 0

    for n, question in enumerate(asked, 1):
        try:
            result = await search(index, question)
        except Exception as exc:                                # noqa: BLE001
            print(f"  {n:2}. ERROR  {question[:48]} ({type(exc).__name__})")
            continue

        answer = result.answer
        # Every labelled line the model was actually shown.
        lines: list[tuple[str, str]] = []
        for hit in result.hits:
            lines.extend(LABELLED.findall(getattr(hit, "text_ts", "") or ""))

        problems = []
        for quote in QUOTED.findall(answer):
            target = words(quote)
            if len(target) < 3:
                continue
            # The labelled line this quote came from, if any.
            best, overlap = None, 0
            for speaker, text in lines:
                shared = len(target & words(text))
                if shared > overlap:
                    best, overlap = speaker, shared
            if not best or overlap < 3:
                unverifiable += 1
                continue
            checked += 1
            # Who does the answer credit, nearest before the quote?
            where = answer.find(quote)
            credits = ATTRIBUTION.findall(answer[:where])
            if not credits:
                continue
            claimed = credits[-1][0]
            claimed = "FaZe Banks" if claimed in ("Banks", "FaZe Banks") \
                else claimed
            if claimed != best:
                problems.append(f'"{quote[:44]}" credited to {claimed}, '
                                f'the line says {best}')

        flag = "FAIL " if problems else "ok   "
        print(f"  {n:2}. {flag}  {question[:52]}")
        for line in problems:
            print(f"        {line}")
            wrong.append(f"{question[:36]}: {line}")
        await asyncio.sleep(args.pause)

    print(f"\n  {checked} quotes traced to a labelled line · "
          f"{len(wrong)} attributed to the wrong person")
    print(f"  {unverifiable} quotes came from unlabelled lines "
          f"(nothing to check against)")
    if wrong:
        print("\n  FAILURES:")
        for line in wrong:
            print(f"     {line}")
    print()
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
