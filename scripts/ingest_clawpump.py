"""Reground the ClawPump support knowledge base.

Ingests data/clawpump_docs.json into the "clawpump" Pinecone namespace.

The namespace is the point. The Bullpen concierge reads the DEFAULT
namespace, and these two bots answer questions that look similar in the
abstract — fees, wallets, launching, order execution — while having
completely different correct answers. Sharing an index without isolation
would let a question about ClawPump's 65% creator fee retrieve Bullpen's
fee documentation, and the model would answer it confidently.

    python scripts/fetch_clawpump_docs.py     # refresh the docs first
    python scripts/ingest_clawpump.py --reset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone  # noqa: E402
from pinecone.errors.exceptions import NotFoundError  # noqa: E402

from app.clawpump import NAMESPACE  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingest import IngestionPipeline  # noqa: E402
from app.schemas import IngestDocument  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCE = "data/clawpump_docs.json"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help=f"clear the {NAMESPACE!r} namespace before ingesting")
    args = ap.parse_args()

    path = ROOT / SOURCE
    if not path.exists():
        print(f"missing {SOURCE} — run scripts/fetch_clawpump_docs.py first")
        return

    raw = json.loads(path.read_text())
    docs = [IngestDocument(**d) for d in raw]
    print(f"  loaded {len(docs)} docs from {SOURCE}")

    if args.reset:
        settings = get_settings()
        index = Pinecone(api_key=settings.pinecone_api_key).Index(
            settings.pinecone_index
        )
        # Scoped to this namespace by name. A delete_all against the default
        # namespace here would wipe the Bullpen concierge, so the namespace
        # is never defaulted or inferred.
        #
        # Pinecone 404s on deleting a namespace that does not exist yet,
        # which is the state of every first run. "Clear it" and "it was
        # already empty" should reach the same place, so that is caught and
        # reported rather than being a crash the operator has to know to
        # expect once.
        try:
            await asyncio.to_thread(
                lambda: index.delete(delete_all=True, namespace=NAMESPACE)
            )
            print(f"  cleared {NAMESPACE!r} namespace")
        except NotFoundError:
            print(f"  {NAMESPACE!r} namespace does not exist yet — nothing to clear")

    pipeline = IngestionPipeline()
    chunks = await pipeline.ingest(docs, namespace=NAMESPACE)
    print(f"\ningested {len(docs)} docs -> {chunks} chunks in {NAMESPACE!r}")


if __name__ == "__main__":
    asyncio.run(main())
