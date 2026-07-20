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

from anthropic import AsyncAnthropic  # noqa: E402

from app.assets_store import AssetStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.podcast import PodcastIndex  # noqa: E402
from app.schemas import Episode  # noqa: E402
from app.summaries import SummaryStore  # noqa: E402
from scripts.extract_assets import extract_episode  # noqa: E402
from scripts.fetch_episodes import OUT, fetch, latest_ids  # noqa: E402

# Cheapest model that reliably does the structured extraction.
ASSET_MODEL = "claude-haiku-4-5"


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
    asset_store = AssetStore()
    anthropic_client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
    added = 0

    for vid in new_ids:
        raw = fetch(vid, cookies)
        if not raw:
            log(f"  {vid}: no captions yet (likely still generating) — skip, "
                f"will retry next run")
            continue
        episode = Episode(**raw)

        # 1. ingest into Pinecone (append — does NOT clear existing episodes).
        #    Do this BEFORE recording the episode as indexed: if ingest fails,
        #    the episode stays out of episodes.json and is retried next run,
        #    rather than being marked done-but-absent forever.
        try:
            windows = await podcast.ingest([episode])
        except Exception as exc:  # noqa: BLE001 — don't abort the whole batch
            log(f"  {vid}: ingest FAILED ({exc}) — will retry next run")
            continue
        log(f"  {vid}: ingested {windows} windows — {episode.title[:55]}")

        # 2. persist to the local record only after a successful ingest.
        indexed.append(raw)
        OUT.write_text(json.dumps(indexed, indent=2, ensure_ascii=False))

        # 3. summarize (idempotent; a summary failure must not lose the ingest)
        try:
            summary = await summaries.summarize(episode)
            await summaries.store(episode, summary)
            log(f"  {vid}: summary stored ({len(summary)} chars)")
        except Exception as exc:  # noqa: BLE001
            log(f"  {vid}: summary FAILED ({exc}) — search still works, "
                f"re-run summarize_episodes.py later")

        # 4. extract the assets discussed, and store them where the DEPLOYED
        #    app reads them. Without this the token dashboard silently goes
        #    stale while search and summaries stay current.
        try:
            hits = await extract_episode(anthropic_client, ASSET_MODEL, raw, False)
            stored = await asset_store.store(vid, episode.title, hits)
            log(f"  {vid}: {stored} asset hits stored")
        except Exception as exc:  # noqa: BLE001
            log(f"  {vid}: asset extraction FAILED ({exc}) — search and "
                f"summary are fine; re-run extract_assets.py --all --store")

        added += 1

    log(f"DONE: added {added} new episode(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
