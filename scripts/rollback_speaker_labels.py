"""Put the passages back the way they were, if the labels go wrong.

    .venv/bin/python scripts/rollback_speaker_labels.py --dry-run
    .venv/bin/python scripts/rollback_speaker_labels.py

Reverting the code is a git push. Reverting the DATA is this, because the
speaker labels live in Pinecone metadata and both branches read the same
index — a rollback that only redeploys the old build would leave every
passage still carrying names.

Nothing is lost either way: text_ts is computed from the transcript, so
the original is recomputed here rather than restored from a backup that
might not exist. Same windows, same overlap, same settings the ingest
used, so what goes back is byte-identical to what was there before.

Run it after rolling back the code, not before — the old build reads the
same passages and is happy with either version, but a labelled passage
under a build whose prompt knows nothing about prefixes is the one
combination nobody has tested.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pinecone import Pinecone  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.podcast import NAMESPACE, _windows  # noqa: E402
from app.schemas import Episode  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
SPEAKER_MAP = ROOT / "data" / "speaker_map.json"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}
    if not SPEAKER_MAP.exists():
        print("  no speaker map — nothing was ever applied")
        return 0
    mapping = json.loads(SPEAKER_MAP.read_text())

    index = Pinecone(api_key=settings.pinecone_api_key).Index(
        settings.pinecone_index)

    todo = []
    for episode_id in mapping:
        episode = Episode(**episodes[episode_id])
        for start_t, _text, stamped in _windows(
                episode.segments, settings.chunk_max_chars,
                overlap_segments=2):
            vector_id = hashlib.sha256(
                f"{episode_id}:{start_t}".encode()).hexdigest()[:32]
            todo.append((vector_id, stamped))

    print(f"\n  {len(todo)} passages across {len(mapping)} episodes would go "
          f"back to unlabelled text")
    if args.dry_run:
        print("  --dry-run: nothing written\n")
        return 0

    done = 0
    for vector_id, stamped in todo:
        await asyncio.to_thread(
            index.update, id=vector_id, namespace=NAMESPACE,
            set_metadata={"text_ts": stamped, "speakers": []})
        done += 1
        if done % 300 == 0:
            print(f"     {done}/{len(todo)} restored", flush=True)

    print(f"\n  {done} passages restored · 0 vectors changed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
