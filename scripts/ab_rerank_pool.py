"""Does widening the rerank pool cost accuracy?

    .venv/bin/python scripts/ab_rerank_pool.py
    .venv/bin/python scripts/ab_rerank_pool.py --pools 12,50 --limit 20

The reranker is handed the top N passages by embedding similarity and
picks six. N is 12, out of twenty-seven thousand. A question whose answer
sits at rank 44 -- "who was the guy who sold all his eth holding", whose
answer is Banks asking and Ansem replying "David Hoffman" -- is invisible
to the component that would recognise it instantly.

Widening N fixes that. The worry is the other direction: more candidates
means more plausible-but-wrong passages reaching the model, and that is
how a line gets credited to the wrong person. So this measures the thing
actually at risk rather than asking whether the answers read well.

Three numbers per pool size:

  misattributed   a quoted fragment credited to a host the transcript
                  says did not say it. This is the one that matters --
                  a false claim about a real person. Borrowed wholesale
                  from verify_attribution.py, same regexes, same rule.

  refused         answers that found nothing. Widening should push this
                  down; if it pushes it down while misattribution holds,
                  the change is free.

  recovered       the near-verbatim probes, where the answer is known to
                  be in the archive and known to rank below 12.

Paced, because fifty back-to-back searches time out against the index
under burst load -- a fact about the index, not a fault.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.podcast import PodcastIndex  # noqa: E402
from scripts.ab_questions import all_questions  # noqa: E402
from scripts.verify_attribution import (  # noqa: E402
    ATTRIBUTION,
    LABELLED,
    QUOTED,
    words,
)

# Questions whose answer is in the archive and ranks below 12, so the
# current setting cannot reach it. Each carries the episode the answer
# lives in, so "recovered" means the right episode, not a lucky refusal.
PROBES = [
    ("who was the guy who sold all his eth holding", "F4OhqZjtVkY"),
    ("which bankless guy sold his ethereum", "F4OhqZjtVkY"),
    ("who capitulated on eth after arguing with ansem", "F4OhqZjtVkY"),
]

# Ordinary questions, to catch a regression somewhere unrelated.
BROAD = [
    "what did ansem say about hyperliquid",
    "what did banks say about his portfolio",
    "what did ansem say about bonk",
    "what did ansem say about solana",
    "what did banks say about the ansem token",
    "what did they say about prediction markets",
    "what did ansem say about zcash",
    "what did banks say about polymarket",
]

REFUSAL_MARKERS = (
    "not in", "couldn't find", "could not find", "don't see", "do not see",
    "not that exact", "not word for word", "no moment", "nothing matching",
    "i'd need more context",
)


def refused(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def misattributions(answer: str, hits) -> list[str]:
    """verify_attribution's rule, applied to one answer."""
    lines: list[tuple[str, str]] = []
    for hit in hits:
        lines.extend(LABELLED.findall(getattr(hit, "text_ts", "") or ""))

    problems = []
    for quote in QUOTED.findall(answer):
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
        credits = ATTRIBUTION.findall(answer[:answer.find(quote)])
        if not credits:
            continue
        claimed = credits[-1][0]
        claimed = "FaZe Banks" if claimed in ("Banks", "FaZe Banks") else claimed
        if claimed != best:
            problems.append(f'"{quote[:40]}" -> {claimed}, line says {best}')
    return problems


async def ask(index: PodcastIndex, question: str, tries: int = 3):
    for attempt in range(tries):
        try:
            return await index.search(question)
        except Exception:                                       # noqa: BLE001
            if attempt + 1 == tries:
                raise
            await asyncio.sleep(6)
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="12,50")
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    # "50+12" means: rerank fifty, and union in the first twelve's own
    # six. Written as one token so the arms stay comparable in the table.
    def spec(text: str) -> tuple[int, int]:
        if "+" in text:
            deep, narrow = text.split("+", 1)
            return int(deep), int(narrow)
        return int(text), 0

    pools = [spec(p) for p in args.pools.split(",")]

    unique = all_questions()
    if args.limit:
        unique = unique[:args.limit]
    probe_total = sum(1 for _q, _e, k in unique if k == "probe")
    absent_total = sum(1 for _q, _e, k in unique if k == "absent")

    print(f"\n  {len(unique)} questions x {len(pools)} pool sizes "
          f"= {len(unique) * len(pools)} answers\n")

    results: dict[int, dict] = {}
    for pool in pools:
        deep, narrow = pool
        name = f"{deep}+{narrow}" if narrow else str(deep)
        # PodcastIndex reads the settings once, at construction.
        get_settings.cache_clear()
        import os
        os.environ["RERANK_CANDIDATES"] = str(deep)
        os.environ["RERANK_NARROW_POOL"] = str(narrow)
        index = PodcastIndex()
        assert (index._settings.rerank_candidates == deep
                and index._settings.rerank_narrow_pool == narrow), (
            "the override did not reach the index -- every number below "
            "would be the same run twice")
        bad, refusals, recovered, errors = [], 0, 0, 0
        invented: list[str] = []
        # Totals alone hid the difference once already: both settings
        # refused ten questions and it read as no change, when a matching
        # count says nothing about which ten.
        outcome: dict[str, str] = {}

        print(f"  ── pool {name}")
        for n, (question, want_episode, kind) in enumerate(unique, 1):
            try:
                result = await ask(index, question)
            except Exception as exc:                            # noqa: BLE001
                errors += 1
                print(f"    {n:3}. ERROR {question[:44]} ({type(exc).__name__})")
                continue
            problems = misattributions(result.answer, result.hits)
            is_refusal = refused(result.answer)
            hit_right = want_episode and any(
                h.episode_id == want_episode for h in result.hits[:3])
            if want_episode and hit_right and not is_refusal:
                recovered += 1
            if is_refusal:
                refusals += 1
            # The direct test of the worry. These topics are not in any
            # transcript -- the terms were searched before the question
            # was written -- so an answer is an answer about nothing.
            if kind == "absent" and not is_refusal:
                invented.append(question)
                print(f"    {n:3}. INVENTED  {question[:46]}")
                print(f"          {result.answer[:150]}")
            outcome[question] = ("refused" if is_refusal else "answered")
            if problems:
                bad.append((question, problems))
                print(f"    {n:3}. MISATTRIB {question[:44]}")
                for p in problems:
                    print(f"          {p}")
            await asyncio.sleep(args.pause)

        results[pool] = {"bad": bad, "refusals": refusals,
                         "recovered": recovered, "errors": errors,
                         "outcome": outcome, "invented": invented}
        print(f"     misattributed {len(bad)} · answered-the-absent "
              f"{len(invented)}/{absent_total} · probes {recovered}"
              f"/{probe_total} · refused {refusals} · errors {errors}\n")

    print("  " + "=" * 68)
    print(f"  {'pool':>6} {'misattrib':>10} {'invented':>10}"
          f" {'probes':>9} {'refused':>9} {'errors':>7}")
    for pool in pools:
        r = results[pool]
        deep, narrow = pool
        name = f"{deep}+{narrow}" if narrow else str(deep)
        print(f"  {name:>6} {len(r['bad']):>10} "
              f"{len(r['invented']):>4}/{absent_total:<5}"
              f" {r['recovered']:>4}/{probe_total:<4} {r['refusals']:>9}"
              f" {r['errors']:>7}")
    print()

    base, wide = pools[0], pools[-1]

    # Which questions actually changed hands, not how many.
    a, b = results[base]["outcome"], results[wide]["outcome"]
    gained = [q for q in a if a[q] == "refused" and b.get(q) == "answered"]
    lost = [q for q in a if a[q] == "answered" and b.get(q) == "refused"]
    if gained:
        print(f"  now answered at pool {wide} ({len(gained)}):")
        for q in gained:
            print(f"     + {q[:64]}")
    if lost:
        print(f"  now refused at pool {wide} ({len(lost)}):")
        for q in lost:
            print(f"     - {q[:64]}")
    if not gained and not lost:
        print("  no question changed between answered and refused.")
    print()

    # The verdict, stated as a rule rather than left to whoever reads
    # the table. Two ways to fail and both block the change.
    worse_names = (len(results[wide]["bad"]) - len(results[base]["bad"]))
    worse_invention = (len(results[wide]["invented"])
                       - len(results[base]["invented"]))
    if worse_names > 0 or worse_invention > 0:
        print(f"  DO NOT SHIP pool {wide}:")
        if worse_names > 0:
            print(f"     {worse_names} more quote(s) credited to the wrong host")
        if worse_invention > 0:
            print(f"     {worse_invention} more answer(s) about topics that are "
                  f"not in any transcript")
    else:
        print(f"  Pool {wide} is safe on both failure modes"
              f" ({-worse_names} fewer misattributions,"
              f" {-worse_invention} fewer inventions).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
