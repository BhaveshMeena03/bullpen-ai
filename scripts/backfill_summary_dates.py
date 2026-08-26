"""Put the air date back on summaries that were written without one.

    .venv/bin/python scripts/backfill_summary_dates.py [--apply]

The episode list on the page sorts by published_at and prints it under each
title. Twenty-one summaries were written before dates were being captured,
so those episodes sorted to the bottom in an arbitrary order and showed no
date at all — while the newer ones, and every X broadcast, showed theirs.
The browse list therefore looked like the catalogue only went back three
weeks.

data/episodes.json has the dates already; a later backfill added them there
and never touched the summaries. Only the metadata is wrong, so this
patches it in place rather than paying to regenerate ~$5 of prose that is
perfectly good.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pinecone import Pinecone  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.summaries import NAMESPACE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the metadata; without this it only reports")
    args = ap.parse_args()

    episodes = {e["episode_id"]: e
                for e in json.loads((ROOT / "data" / "episodes.json").read_text())}
    settings = get_settings()
    index = Pinecone(api_key=settings.pinecone_api_key).Index(
        settings.pinecone_index)

    ids = [f"summary-{k}" for k in episodes]
    fetched = index.fetch(ids=ids, namespace=NAMESPACE)
    vectors = fetched.vectors if hasattr(fetched, "vectors") else fetched["vectors"]

    fixed = 0
    for episode_id, episode in episodes.items():
        vector = vectors.get(f"summary-{episode_id}")
        if not vector:
            continue                      # no summary yet; not this script's job
        meta = dict((vector.get("metadata") if isinstance(vector, dict)
                     else vector.metadata) or {})
        date = episode.get("published_at")
        if not date or meta.get("published_at"):
            continue
        print(f"  + {date}  {episode.get('title', '')[:52]}")
        fixed += 1
        if args.apply:
            index.update(id=f"summary-{episode_id}", namespace=NAMESPACE,
                         set_metadata={"published_at": date})

    print(f"\n  {fixed} summary/summaries missing a date")
    if not args.apply:
        print("  reporting only — pass --apply to write")


if __name__ == "__main__":
    main()
