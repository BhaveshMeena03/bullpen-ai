"""Reground the concierge knowledge base.

Ingests the ecosystem docs (data/seed.json — Ansem/Banks/Market Bubble,
$ANSEM, crypto basics, scam safety) plus the official Bullpen docs
(data/bullpen_docs.json) into the concierge's Pinecone namespace.

With --reset it clears that namespace first, so docs that were removed or
renamed don't linger as orphaned vectors. The concierge lives in the DEFAULT
namespace; podcast / summaries / assets each use their own, so a reset here
never touches them.

    python scripts/fetch_bullpen_docs.py       # refresh official docs first
    python scripts/ingest_concierge.py --reset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.ingest import IngestionPipeline  # noqa: E402
from app.schemas import IngestDocument  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FILES = ["data/seed.json", "data/bullpen_docs.json"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="clear the concierge namespace before ingesting")
    args = ap.parse_args()

    docs: list[IngestDocument] = []
    for rel in FILES:
        path = ROOT / rel
        if not path.exists():
            print(f"  missing {rel} — skipping")
            continue
        raw = json.loads(path.read_text())
        docs.extend(IngestDocument(**d) for d in raw)
        print(f"  loaded {len(raw):3d} docs from {rel}")

    if not docs:
        print("nothing to ingest")
        return

    if args.reset:
        settings = get_settings()
        index = Pinecone(api_key=settings.pinecone_api_key).Index(
            settings.pinecone_index
        )
        await asyncio.to_thread(
            lambda: index.delete(delete_all=True, namespace="")
        )
        print("  cleared concierge (default) namespace")

    pipeline = IngestionPipeline()
    chunks = await pipeline.ingest(docs)
    print(f"\ningested {len(docs)} docs -> {chunks} chunks")


if __name__ == "__main__":
    asyncio.run(main())
