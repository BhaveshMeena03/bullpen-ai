"""Weekly sync: add any new Market Bubble episodes to the live index.

One command, idempotent and incremental:
  1. list the channel's recent videos (needs browser cookies for YouTube)
  2. skip episodes already in data/episodes.json
  3. for each NEW one: fetch captions -> append to episodes.json
     -> ingest into Pinecone (append, no re-embedding of old episodes)
     -> generate its summary
  4. log what happened

Safe to run repeatedly — already-indexed episodes are skipped, so a
scheduled run that finds nothing simply does nothing.

    .venv/bin/python scripts/sync_latest.py                 # cookies from chrome
    .venv/bin/python scripts/sync_latest.py --cookies safari
    .venv/bin/python scripts/sync_latest.py --check 8       # scan newest 8
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.podcast import PodcastIndex  # noqa: E402
from app.schemas import Episode  # noqa: E402
from app.summaries import SummaryStore  # noqa: E402
from scripts.fetch_episodes import OUT, fetch, latest_ids  # noqa: E402


def log(msg: str) -> None:
    # Timestamped so the scheduler's log file is readable after the fact.
    import datetime
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def load_indexed() -> list[dict]:
    if OUT.exists():
        try:
            return json.loads(OUT.read_text())
        except json.JSONDecodeError:
            return []
    return []


async def main(argv: list[str]) -> int:
    cookies = "chrome"
    check = 8
    if "--cookies" in argv:
        cookies = argv[argv.index("--cookies") + 1]
    if "--check" in argv:
        check = int(argv[argv.index("--check") + 1])

    indexed = load_indexed()
    known = {e["episode_id"] for e in indexed}
    log(f"{len(known)} episodes already indexed; scanning newest {check}...")

    ids = latest_ids(check, cookies)
    if not ids:
        log("could not list channel (cookies expired? YouTube block?) — aborting")
        return 1

    new_ids = [v for v in ids if v not in known]
    if not new_ids:
        log("no new episodes. Nothing to do.")
        return 0

    log(f"{len(new_ids)} new episode(s): {new_ids}")
    podcast = PodcastIndex()
    summaries = SummaryStore()
    added = 0

    for vid in new_ids:
        raw = fetch(vid, cookies)
        if not raw:
            log(f"  {vid}: no captions yet (likely still generating) — skip, "
                f"will retry next run")
            continue
        episode = Episode(**raw)

        # 1. append to the local record (source of truth for 'what's indexed')
        indexed.append(raw)
        OUT.write_text(json.dumps(indexed, indent=2, ensure_ascii=False))

        # 2. ingest into Pinecone (append — does NOT clear existing episodes)
        windows = await podcast.ingest([episode])
        log(f"  {vid}: ingested {windows} windows — {episode.title[:55]}")

        # 3. summarize (idempotent)
        try:
            summary = await summaries.summarize(episode)
            await summaries.store(episode, summary)
            log(f"  {vid}: summary stored ({len(summary)} chars)")
        except Exception as exc:  # summary failure shouldn't lose the ingest
            log(f"  {vid}: summary FAILED ({exc}) — search still works, "
                f"re-run summarize_episodes.py later")
        added += 1

    log(f"DONE: added {added} new episode(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
