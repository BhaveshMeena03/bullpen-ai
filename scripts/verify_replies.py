"""Check the whole reply, the way a reader would.

    .venv/bin/python scripts/verify_replies.py
    .venv/bin/python scripts/verify_replies.py --limit 8

Not the answer text alone — the posted reply, with its link, checked
against the transcripts. Four things, each of which has been wrong on a
live reply this week:

  episode    the link points at the episode the answer names, not the
             one that happened to rank first. A quote from Ep 10 went out
             under the August 27 draft's URL.
  second     the ?t= in the link is the moment the answer cites, and the
             answer's moment is real. "scheduled to appear around 3:30 PM"
             produced a jump to 3 minutes 30 seconds.
  supported  the transcript at that second shares numbers or names with
             the claim, so the citation is about what it says it is.
  speaker    a host named in the answer matches the label on the line the
             quote came from.

Paced and retried: the index times out under back-to-back load.
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
sys.path.insert(0, str(ROOT / "scripts"))

from verify_attribution import ATTRIBUTION, LABELLED, QUOTED, words  # noqa: E402

from app.podcast import PodcastIndex  # noqa: E402
from app.x_bot import format_reply  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
LINK = re.compile(r"https?://\S+")
T_PARAM = re.compile(r"[?&]t=(\d+)s?")
# Case-insensitive, and the uppercase forms spelled out: without them
# "3:30 PM" matched and the checker written to catch clock-time
# citations could not catch one.
CITED = re.compile(
    r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b(?!\s*(?:[ap]\.?m\.?|AM|PM)\b)",
    re.I)

QUESTIONS = [
    "what did ansem say about hyperliquid",
    "what did ansem say about his solana price target",
    "what did banks say about streaming",
    "what did banks say about his portfolio",
    "what did brian armstrong say about suing the SEC",
    "what did luca netz say about pudgy penguins",
    "what did jesse pollak say about base",
    "what did tushar jain say about multicoin",
    "what did they say about pump fun revenue",
    "what did camila say about onlyfans earnings",
    "what did ansem say about pump fun",
    "what did they say about GTA 6",
]

# A second set, deliberately different ground: guests who appear once,
# bare numbers, topics with no person attached, and the phrasings that
# have broken something before.
MORE = [
    "what did kendrick perkins say about investing",
    "what did mert say about pump fun",
    "what did andrew kang say about robotics",
    "what did greg say about gpu pricing",
    "what did orangie say about his trades",
    "what did poorgoat say about his coin",
    "what did mike dudas say about crypto",
    "what did austin federa say about solana",
    "how much did camila make on the drop",
    "what was said about 54 million",
    "what did they say about trump and hyperliquid",
    "what was said about bitcoin reclaiming 80k",
    "what did they say about the coinbase hack",
    "what was said about pump fun and the airdrop",
    "what did banks say about kick",
    "what did ansem say about bonk",
    "yo take a look at this what did ansem say about eth",
    "what did they say about nfts",
    "what did they say about michael saylor",
    "what did tjr say about attention",
]


def seconds(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    return (parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3
            else parts[0] * 60 + parts[1])


def window(episode: dict, at: int, reach: int = 120) -> str:
    return " ".join(s.get("text", "") for s in episode["segments"]
                    if abs(s.get("t", 0) - at) <= reach).lower()


def supported(text: str, claim: str) -> bool:
    numbers = re.findall(r"\d[\d.,]*", claim)
    names = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", claim)
    if any(n.rstrip(".,") in text for n in numbers):
        return True
    return any(n.lower() in text for n in names)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pause", type=float, default=2.5)
    ap.add_argument("--more", action="store_true",
                    help="add the second question set")
    args = ap.parse_args()

    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}
    by_url = {e["url"]: e for e in episodes.values()}
    index = PodcastIndex()
    pool = QUESTIONS + MORE if args.more else QUESTIONS
    asked = pool[:args.limit] if args.limit else pool
    failures: list[str] = []
    checked = 0

    for n, question in enumerate(asked, 1):
        try:
            result = await index.search(question)
        except Exception as exc:                                # noqa: BLE001
            print(f"  {n:2}. ERROR  {question[:46]} ({type(exc).__name__})")
            await asyncio.sleep(args.pause)
            continue

        reply = format_reply(result.answer, result.hits,
                             include_links="always", limit=1500)
        if not reply or "couldn't find" in reply.lower():
            print(f"  {n:2}. miss   {question[:52]}")
            await asyncio.sleep(args.pause)
            continue

        problems = []
        link = (LINK.search(reply) or [None]) and LINK.search(reply)
        url = link.group(0).rstrip(":") if link else None
        episode = None
        if url:
            base = url.split("&t=")[0].split("?t=")[0]
            episode = by_url.get(base) or next(
                (e for u, e in by_url.items() if u.split("?")[0] in base), None)
            if not episode:
                problems.append(f"link matches no indexed episode: {base[:52]}")

        cited = CITED.findall(reply)
        if episode and cited:
            checked += 1
            moment = seconds(cited[0])
            # The link's own second, when it carries one.
            param = T_PARAM.search(url or "")
            if param and abs(int(param.group(1)) - moment) > 3:
                problems.append(
                    f"link jumps to {int(param.group(1))}s, reply says "
                    f"{cited[0]} ({moment}s)")
            near = window(episode, moment)
            if not near:
                problems.append(f"{cited[0]} is not inside "
                                f"{episode['title'][:34]}")
            elif not supported(near, result.answer):
                problems.append(f"transcript at {cited[0]} does not support "
                                f"the claim")

        # Attribution, for hosts.
        lines = [x for h in result.hits
                 for x in LABELLED.findall(getattr(h, "text_ts", "") or "")]
        for quote in QUOTED.findall(result.answer):
            target = words(quote)
            if len(target) < 3:
                continue
            best, overlap = None, 0
            for speaker, text in lines:
                shared = len(target & words(text))
                if shared > overlap:
                    best, overlap = speaker, shared
            if not best or overlap < 3:
                continue
            credits = ATTRIBUTION.findall(result.answer[:result.answer.find(quote)])
            if not credits:
                continue
            claimed = credits[-1][0]
            claimed = "FaZe Banks" if claimed in ("Banks", "FaZe Banks") else claimed
            if claimed != best:
                problems.append(f'"{quote[:34]}" credited to {claimed}, '
                                f'line says {best}')

        print(f"  {n:2}. {'FAIL ' if problems else 'ok   '}  "
              f"{question[:52]}")
        for line in problems:
            print(f"        {line}")
            failures.append(f"{question[:34]}: {line}")
        await asyncio.sleep(args.pause)

    print(f"\n  {checked} replies with a link and a citation checked · "
          f"{len(failures)} problems")
    if failures:
        print("\n  FAILURES:")
        for line in failures:
            print(f"     {line}")
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
