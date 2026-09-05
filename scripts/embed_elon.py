"""Put the Musk transcripts into their own namespace, so the page can search.

    .venv/bin/python scripts/embed_elon.py --dry-run
    .venv/bin/python scripts/embed_elon.py

Separate from the ingest because they fail differently and cost
differently. Transcription is hours of local CPU and is resumable for
free; embedding spends money at Voyage and quota at Pinecone, and the
Pinecone egress limit has already taken search down once on this project.
So this step is explicit, says what it will do before it does it, and can
be asked to do nothing.

Writes to the `elon` namespace and nowhere else. The Market Bubble vectors
live in `podcast` and the two never meet -- @mbubbleSearch's whole standing
is that it answers from that broadcast, and one reply about the show
sourced from a Tesla interview would prove it cannot tell them apart.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.podcast import PodcastIndex                    # noqa: E402
from app.schemas import Episode                         # noqa: E402

NAMESPACE = "elon"
EPISODES = ROOT / "data" / "elon_episodes.json"


def load() -> list[dict]:
    if not EPISODES.exists():
        raise SystemExit(f"  nothing to embed: {EPISODES} does not exist")
    return json.loads(EPISODES.read_text())


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be embedded and stop")
    ap.add_argument("--only", help="one episode id")
    args = ap.parse_args()

    rows = load()
    if args.only:
        rows = [e for e in rows if e["episode_id"] == args.only]
    if not rows:
        raise SystemExit("  no episodes matched")

    hours = sum(max((s["t"] for s in e["segments"]), default=0)
                for e in rows) / 3600
    lines = sum(len(e["segments"]) for e in rows)
    print(f"\n  {len(rows)} recordings, {hours:.1f} hours, {lines:,} lines")
    for e in rows:
        span = max((s["t"] for s in e["segments"]), default=0) / 3600
        print(f"    {e.get('published_at','')}  {span:4.1f}h  {e['title'][:52]}")
    print(f"\n  destination: Pinecone namespace {NAMESPACE!r}")

    if args.dry_run:
        print("  dry run — nothing sent\n")
        return 0

    index = PodcastIndex(namespace=NAMESPACE)

    # Refuse to write into the broadcast's namespace, whatever anyone has
    # edited. This is the failure that has no recovery: mixed vectors
    # cannot be told apart afterwards without re-embedding everything.
    if index._namespace != NAMESPACE:
        raise SystemExit(f"  refusing: index is pointed at "
                         f"{index._namespace!r}, not {NAMESPACE!r}")

    # Already in? Appending the same windows twice doubles every passage
    # and there is no way to tell the copies apart later.
    for episode in rows:
        existing = index.index.query(
            vector=[0.0] * 1024, top_k=1, namespace=NAMESPACE,
            include_metadata=False,
            filter={"episode_id": {"$eq": episode["episode_id"]}})
        if getattr(existing, "matches", []):
            print(f"  {episode['episode_id']} already has vectors — skipping")
            continue
        print(f"  embedding {episode['title'][:50]}…", flush=True)
        windows = await index.ingest([Episode(**{
            k: v for k, v in episode.items() if k != "channel"})])
        print(f"     {windows} searchable passages", flush=True)

    print(f"\n  done — {NAMESPACE!r} namespace\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
