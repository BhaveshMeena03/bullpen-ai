"""Put the speaker names into the passages, without touching a vector.

    .venv/bin/python scripts/apply_speaker_labels.py --dry-run
    .venv/bin/python scripts/apply_speaker_labels.py

Two things get written, and neither is an embedding:

  text_ts    the timestamped copy of a passage, which is what the model
             actually reads when it writes an answer. Every line gains a
             name where the map has one: "[12:02] FaZe Banks: I put close
             to seven figures in Hyperliquid". This is what stops the
             model borrowing a subject from the question, which is how
             somebody else's portfolio was reported as Banks losing
             $254,000.

  speakers   which hosts talk in the passage, so a question naming one
             can be narrowed to passages where they actually speak. The
             name itself cannot go in the term index: that index drops
             anything appearing in more than 60 passages, and Ansem
             speaks in most of the archive, so forcing him in would flood
             the rerank pool with thousands of undiscriminating hits.

The embedded `text` is untouched, deliberately. It is the same decision
the timestamps got, and the reason is in the _windows docstring: keeping
the embedded text identical means the ranking after this change is
provably the same ranking as before it. That is a promise worth more than
any answer this improves.

Metadata-only updates, keyed by the id ingest computes from
sha256(episode_id:start_seconds) — the same passage, edited in place.
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
from app.podcast import NAMESPACE as PODCAST_NS  # noqa: E402
from app.podcast import _timestamp, _windows  # noqa: E402
from app.schemas import Episode  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
SPEAKER_MAP = ROOT / "data" / "speaker_map.json"


def vector_id(episode_id: str, start_seconds: float) -> str:
    return hashlib.sha256(
        f"{episode_id}:{start_seconds}".encode()).hexdigest()[:32]


def stamped_with_speakers(episode: Episode, speakers: dict[str, str],
                          start: float, end: float) -> tuple[str, list[str]]:
    """The passage's timestamped lines, with a name on the ones we know.

    Segments are numbered as the fingerprinting numbered them: non-empty
    text only, in order. A line whose speaker is unknown keeps exactly
    the shape it had, so an unlabelled passage is byte-identical to what
    is already stored and costs no write.
    """
    lines, present = [], []
    numbered = [s for s in episode.segments if (s.text or "").strip()]
    for i, segment in enumerate(numbered):
        if not (start <= segment.t < end):
            continue
        who = speakers.get(str(i))
        stamp = _timestamp(segment.t)
        if who:
            lines.append(f"[{stamp}] {who}: {segment.text.strip()}")
            if who not in present:
                present.append(who)
        else:
            lines.append(f"[{stamp}] {segment.text.strip()}")
    return "\n".join(lines), present


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change and write nothing")
    ap.add_argument("--limit", type=int, help="only this many episodes")
    args = ap.parse_args()

    settings = get_settings()
    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}
    mapping = json.loads(SPEAKER_MAP.read_text())
    index = Pinecone(api_key=settings.pinecone_api_key).Index(
        settings.pinecone_index)

    todo = list(mapping)[:args.limit] if args.limit else list(mapping)
    updates: list[tuple[str, str, list[str]]] = []
    named = plain = 0

    for episode_id in todo:
        episode = Episode(**episodes[episode_id])
        speakers = mapping[episode_id]
        windows = list(_windows(episode.segments,
                                settings.chunk_max_chars, overlap_segments=2))
        for n, (start_t, _text, _stamped) in enumerate(windows):
            end_t = (windows[n + 1][0] if n + 1 < len(windows)
                     else float("inf"))
            text_ts, present = stamped_with_speakers(
                episode, speakers, start_t, end_t)
            if not text_ts:
                continue
            if present:
                named += 1
                updates.append((vector_id(episode_id, start_t),
                                text_ts, present))
            else:
                plain += 1

    print(f"\n  {len(todo)} episodes · {named} passages gain a speaker · "
          f"{plain} have none and are left alone")
    if updates:
        _, sample, who = updates[0]
        print(f"\n  sample ({', '.join(who)}):")
        for line in sample.splitlines()[:3]:
            print(f"     {line[:88]}")

    if args.dry_run:
        print("\n  --dry-run: nothing written. No embedding is touched "
              "either way.\n")
        return 0

    done = 0
    for vid, text_ts, present in updates:
        # update(), not upsert(): upsert without values would rewrite the
        # vector, and the entire promise of this change is that the
        # vectors are the ones that were there before.
        await asyncio.to_thread(
            index.update, id=vid, namespace=PODCAST_NS,
            set_metadata={"text_ts": text_ts, "speakers": present})
        done += 1
        if done % 200 == 0:
            print(f"     {done}/{len(updates)} passages updated", flush=True)

    print(f"\n  {done} passages updated · 0 vectors changed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
