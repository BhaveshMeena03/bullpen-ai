"""Generate and store episode summaries (one Claude call per episode, once).

Idempotent: episodes that already have a stored summary are skipped, so
re-running after adding new episodes only pays for the new ones.

    .venv/bin/python scripts/summarize_episodes.py            # all missing
    .venv/bin/python scripts/summarize_episodes.py --limit 2  # first N missing
    .venv/bin/python scripts/summarize_episodes.py VIDEO_ID   # specific one
    .venv/bin/python scripts/summarize_episodes.py --force VIDEO_ID  # redo
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import Episode  # noqa: E402
from app.summaries import SummaryStore  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "episodes.json"


def log(msg: str) -> None:
    print(msg, flush=True)


async def main(argv: list[str]) -> int:
    force = "--force" in argv
    argv = [a for a in argv if a != "--force"]
    limit = None
    if argv[:1] == ["--limit"]:
        limit = int(argv[1])
        argv = argv[2:]
    only_ids = set(argv)

    episodes = [Episode(**e) for e in json.loads(DATA.read_text())]
    if only_ids:
        episodes = [e for e in episodes if e.episode_id in only_ids]

    store = SummaryStore()
    done = 0
    for episode in episodes:
        if limit is not None and done >= limit:
            break
        if not force and await store.exists(episode.episode_id):
            log(f"  = {episode.episode_id} already summarized — skip")
            continue
        started = time.monotonic()
        log(f"  … summarizing {episode.episode_id} "
            f"({len(episode.segments)} segments) — {episode.title[:55]}")
        summary = await store.summarize(episode)
        await store.store(episode, summary)
        log(f"  ✓ {episode.episode_id} stored "
            f"({len(summary)} chars, {int(time.monotonic() - started)}s)")
        done += 1

    log(f"DONE: {done} new summaries")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
