"""Drop summary topic lines that point at the wrong moment.

    .venv/bin/python scripts/verify_summaries.py            # report only
    .venv/bin/python scripts/verify_summaries.py --apply    # rewrite them
    .venv/bin/python scripts/verify_summaries.py --episode 11

Every topic line in a stored summary carries a timestamp, and the reply to
"summarize episode 15" is mostly those lines. They are model output and
nothing checked them.

What checking found, after two false starts worth recording:

Word overlap between the topic line and the transcript does not work. The
line is a paraphrase and the transcript is speech; a plainly wrong line
about a Moderna stock surge scored 0.42 while a plainly correct one about
two people meeting scored 0.15.

Asking a model "is this happening at this moment" does not work either. A
topic line marks where a topic BEGINS, so a line that introduces a guest
who then talks for ten minutes fails a question about the next ninety
seconds. That measurement said 37% were wrong.

Asking "does this topic begin in this stretch", over the window a topic
actually opens in, says 8%. That is the number this script acts on, and
the difference between the two is the difference between regenerating
every episode and deleting a few lines.

The timestamps themselves are sound: 73% land exactly on a transcript
line and 27% within ten seconds, so the model is copying real markers and
occasionally attaching one to the wrong topic.
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

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.schemas import Episode  # noqa: E402
from app.summaries import SummaryStore  # noqa: E402
from app.x_bot import _seconds, episode_number  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"

# A topic opens at its timestamp and develops after it, so the window runs
# forward. The short reach backwards catches a line placed a beat late.
LOOK_BACK = 45
LOOK_FORWARD = 300

TOPIC_LINE = re.compile(r"^(\s*[-*]?\s*)\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)$")

JUDGE = """A podcast summary says a topic BEGINS at a given moment. Below is \
the transcript from 45 seconds before that moment to five minutes after.

Does that topic begin somewhere in this stretch? It is a paraphrase, so the \
words will differ, and a topic may be introduced briefly and developed later. \
Answer NO only if this stretch is clearly about something else.

Answer with one word: YES or NO.

TOPIC: {topic}

TRANSCRIPT:
{window}"""


def window(episode: dict, seconds: int) -> str:
    return " ".join(
        s.get("text", "") for s in episode["segments"]
        if -LOOK_BACK <= s.get("t", 0) - seconds <= LOOK_FORWARD
    )[:5000]


async def supported(client, model, gate, episode: dict,
                    stamp: str, topic: str) -> bool:
    async with gate:
        try:
            seconds = _seconds(stamp)
        except ValueError:
            return False
        response = await client.messages.create(
            model=model, max_tokens=8,
            messages=[{"role": "user", "content": JUDGE.format(
                topic=topic, window=window(episode, seconds))}],
        )
        said = "".join(b.text for b in response.content if b.type == "text")
        return said.strip().upper().startswith("YES")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the stored summaries; without it, reports")
    ap.add_argument("--episode", type=int,
                    help="only this episode number")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    gate = asyncio.Semaphore(args.concurrency)
    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}

    store = SummaryStore()
    summaries = await store.list_all()
    if args.episode is not None:
        summaries = [s for s in summaries
                     if episode_number(s.get("title", "")) == args.episode]

    checked = dropped = 0
    rewritten: list[tuple[dict, str]] = []

    for summary in summaries:
        episode = episodes.get(summary.get("episode_id"))
        if not episode:
            print(f"  no transcript for {summary.get('title','?')[:44]} "
                  f"— left alone")
            continue

        lines = summary.get("summary", "").splitlines()
        verdicts = await asyncio.gather(*[
            supported(client, settings.search_model, gate, episode,
                      m.group(2), m.group(3))
            for m in (TOPIC_LINE.match(ln) for ln in lines) if m
        ])

        kept, verdict_iter, removed = [], iter(verdicts), []
        for line in lines:
            found = TOPIC_LINE.match(line)
            if not found:
                kept.append(line)
                continue
            checked += 1
            if next(verdict_iter):
                kept.append(line)
            else:
                removed.append(f"[{found.group(2)}] {found.group(3)[:70]}")

        if removed:
            dropped += len(removed)
            print(f"\n  {summary.get('title','?')[:56]}")
            for line in removed:
                print(f"     drop {line}")
            rewritten.append((summary, "\n".join(kept)))

    print(f"\n  {checked} topic lines checked · {dropped} point elsewhere "
          f"({100 * dropped / max(checked, 1):.0f}%)")
    print(f"  judged with {settings.search_model}, "
          f"about ${checked * 0.0022:.2f}")

    if not args.apply:
        print("\n  Nothing was written. Re-run with --apply to rewrite them.\n")
        return 0

    for summary, text in rewritten:
        episode = episodes[summary["episode_id"]]
        await store.store(Episode(**episode), text)
        print(f"  rewrote {summary.get('title','?')[:52]}")
    print(f"\n  {len(rewritten)} summaries rewritten\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
