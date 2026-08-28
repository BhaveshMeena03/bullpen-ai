"""Take a live broadcast from a URL to searchable, in one command.

    .venv/bin/python scripts/add_broadcast.py https://x.com/MarketBubble/status/...
    .venv/bin/python scripts/add_broadcast.py URL --date 2026-09-03

Roughly half of every show is only on X. The episode goes out live, and the
YouTube upload lands a day later with about a third of it cut — so the window
where this index has something nobody else does is the day in between, and it
is worth being able to use that window without remembering four commands in
the right order.

What it does:

    transcribe   locally, on this machine, with MLX Whisper
    ingest       appends to the index — does NOT clear and rebuild
    summarize    so the episode appears when browsing, not only in search
    terms        rebuilds the exact-token index, so names in the new
                 episode are findable by name and not only by meaning
    highlights   refreshes the pool the bot draws unprompted answers from
    verify       asks the live index a question about it and shows the answer

Appending matters. scripts/ingest_episodes.py clears the namespace and
re-embeds all 2,538 windows to add one episode, which costs real money and
leaves the search returning nothing while it runs. This adds only the new
one.

The YouTube version needs nothing from you: a GitHub Action checks twice a
day and emails when it finds one, and scripts/sync_latest.py finishes the
job. Citations move to YouTube on their own wherever both cover the same
moment, because a YouTube link can jump to the second and an X one cannot.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.episode_store import load  # noqa: E402
from app.podcast import PodcastIndex  # noqa: E402
from app.schemas import Episode  # noqa: E402
from app.summaries import SummaryStore  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"


def step(n: int, of: int, what: str) -> None:
    print(f"\n{'─' * 62}\n  {n}/{of}  {what}\n{'─' * 62}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="the x.com broadcast or clip URL")
    ap.add_argument("--date", help="YYYY-MM-DD if X reports the wrong one")
    ap.add_argument("--model", default="turbo", choices=["turbo", "large-v3"])
    ap.add_argument("--skip-highlights", action="store_true",
                    help="leave the unprompted-answer pool as it is")
    args = ap.parse_args()

    began = time.time()
    before = {e["episode_id"] for e in load(EPISODES)}

    # 1 — transcribe ---------------------------------------------------------
    step(1, 6, "transcribing (local, no API — this is the slow part)")
    cmd = [sys.executable, "scripts/transcribe_x_broadcast.py", args.url,
           "--model", args.model]
    if args.date:
        cmd += ["--date", args.date]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        print("\n  transcription failed — nothing was indexed")
        return 1

    episodes = load(EPISODES)
    fresh = [e for e in episodes if e["episode_id"] not in before]
    if not fresh:
        # The transcriber skips a broadcast that overlaps one already held,
        # which is a correct outcome and not a failure.
        print("\n  nothing new — already indexed, or skipped as a duplicate")
        return 0

    # 2 — ingest -------------------------------------------------------------
    step(2, 6, f"indexing {len(fresh)} episode(s) — appending, not rebuilding")
    index = PodcastIndex()
    parsed = [Episode(**e) for e in fresh]
    windows = await index.ingest(parsed)
    print(f"  {windows} searchable passages added")

    # 3 — summarize ----------------------------------------------------------
    step(3, 6, "summarising, so it shows up when browsing")
    summaries = SummaryStore()
    for episode in parsed:
        try:
            await summaries.store(episode, await summaries.summarize(episode))
            print(f"  summarised {episode.title[:52]}")
        except Exception as exc:                              # noqa: BLE001
            # Not fatal: the episode is already searchable, which is the
            # part that mattered.
            print(f"  summary failed ({exc}) — searchable anyway, rerun later")

    # 4 — the exact-token index ----------------------------------------------
    # Rebuilt here rather than remembered later: a stale index simply has no
    # entry for the new episode, so a question about it silently loses the
    # exact-name matching and nobody finds out.
    step(4, 6, "rebuilding the exact-token index")
    subprocess.run([sys.executable, "scripts/build_term_index.py"], cwd=ROOT)

    # 5 — highlights ---------------------------------------------------------
    step(5, 6, "refreshing the pool the bot answers from unprompted")
    if args.skip_highlights:
        print("  skipped")
    else:
        subprocess.run([sys.executable, "scripts/make_highlights.py"], cwd=ROOT)

    # 6 — prove it -----------------------------------------------------------
    step(6, 6, "asking the live index about it")
    for episode in parsed:
        # Named guests are what people actually search for, and the title is
        # where they are named.
        who = episode.title.split(":")[0].replace("LIVE W/", "").strip()
        question = f"what did {who.split('&')[0].strip().lower()} say"
        result = await index.search(question)
        print(f"\n  Q: {question}")
        print(f"  A: {result.answer[:200]}")
        found = any(h.episode_id == episode.episode_id for h in result.hits)
        print(f"     {'cites the new episode' if found else 'NOT cited yet — '
                     'reranking may just prefer an older take on the same topic'}")

    total = len(load(EPISODES))
    print(f"\n{'─' * 62}")
    print(f"  done in {(time.time() - began) / 60:.0f} min · "
          f"{total} episodes indexed")
    print("  the bot needs no restart — it reads the same index.")
    print(f"{'─' * 62}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
