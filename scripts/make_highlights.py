"""Build a pool of striking moments the bot can offer unprompted.

    .venv/bin/python scripts/make_highlights.py
    .venv/bin/python scripts/make_highlights.py --per-episode 3

People tag the account to say "very cool" far more often than to ask it
anything. Silence is the safe answer and a wasted moment: someone is looking
at the account, and the one thing that would convince them is a demonstration
rather than a thank-you.

So each compliment gets a different genuinely interesting moment from the
archive. Generated once, here, rather than per reply — a model call at reply
time costs money, adds latency, and can produce a dud in public. Picking from
a pool that was reviewed beforehand cannot.

Written to data/highlights.json. Costs roughly $0.008 per episode, so about
$0.25 for the whole catalogue, and only needs rerunning when episodes are
added.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.podcast import _timestamp  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
OUT = ROOT / "data" / "highlights.json"

PROMPT = """\
Below is a transcript of one episode of a crypto podcast, with a timestamp \
on every line.

Pick the {n} most striking, concrete moments in it — the things someone \
would repeat to a friend. A specific number, a real decision, a claim that \
sounds wrong until you hear the reason. Not general discussion, not \
introductions, not "they talked about X".

For each, write ONE sentence naming WHO said it and what, followed by the \
timestamp of the line you took it from, exactly as it appears in brackets.

Two hard rules, because these get posted publicly with nobody checking them \
first:

1. Name the person. If the transcript does not make clear who is speaking, \
skip that moment entirely. "Someone claims" is not worth posting — it reads \
as a rumour and it is the archive's whole value to know who said what.

2. Quote numbers exactly as they were said. Never compute one. Do not turn \
"450 to 500" into a percentage, do not convert, do not total anything up. An \
arithmetic slip becomes a false claim under someone's name, and one already \
did: "$450 to $500 — roughly 50%", which is 11%.

Format, one per line, nothing else:
<sentence> | <timestamp>

If fewer than {n} moments in this transcript genuinely qualify, return fewer. \
A thin moment is worse than no moment.
"""


async def highlights_for(client, model, episode: dict, n: int) -> list[dict]:
    lines = [f"[{_timestamp(s['t'])}] {s['text'].strip()}"
             for s in episode["segments"] if s.get("text", "").strip()]
    # Every third line, capped: enough of the shape of the episode to judge
    # what stands out, without paying to send four hours of speech.
    sampled = "\n".join(lines[::3][:1400])

    response = await client.messages.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content":
                   f"{PROMPT.format(n=n)}\n\n<transcript>\n{sampled}\n</transcript>"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")

    out = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        said, _, stamp = line.rpartition("|")
        said, stamp = said.strip(" -–—"), stamp.strip().strip("[]")
        if len(said) < 40 or not stamp:
            continue
        out.append({
            "episode_id": episode["episode_id"],
            "title": episode["title"],
            "timestamp": stamp,
            "text": said,
        })
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-episode", type=int, default=2)
    args = ap.parse_args()

    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    episodes = json.loads(EPISODES.read_text())

    pool: list[dict] = []
    for i, episode in enumerate(episodes, 1):
        try:
            found = await highlights_for(client, settings.summary_model,
                                         episode, args.per_episode)
        except Exception as exc:                              # noqa: BLE001
            print(f"  [{i}/{len(episodes)}] FAILED {episode['title'][:40]}: {exc}")
            continue
        pool.extend(found)
        print(f"  [{i}/{len(episodes)}] {len(found)} from "
              f"{episode['title'][:44]}")

    OUT.write_text(json.dumps(pool, ensure_ascii=False, indent=2))
    print(f"\n  {len(pool)} highlights -> {OUT.relative_to(ROOT)}")
    print("  Read them before enabling: these get posted unprompted, so a "
          "weak one is worse than none.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
