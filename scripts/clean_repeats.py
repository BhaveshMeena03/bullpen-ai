"""Strip Whisper repetition loops from data/episodes.json.

    .venv/bin/python scripts/clean_repeats.py [--apply]

Whisper sometimes gets stuck and emits one short line dozens of times in a
row. Chunking packs consecutive segments together, so a long enough run
becomes a window that is almost entirely one repeated phrase — an index
slot holding nothing, which retrieval can still return instead of real
content.

New transcriptions are cleaned as they are made. This is for what was
already ingested.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.captions import collapse_repeats  # noqa: E402
from app.episode_store import merge as merge_episodes  # noqa: E402

OUT = ROOT / "data" / "episodes.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(OUT))
    ap.add_argument("--apply", action="store_true",
                    help="write the cleaned file; without this it only reports")
    args = ap.parse_args()

    path = Path(args.file)
    episodes = json.loads(path.read_text())
    total_before = total_after = 0
    for episode in episodes:
        before = episode["segments"]
        after = collapse_repeats(before)
        total_before += len(before)
        total_after += len(after)
        if len(after) < len(before):
            print(f"  -{len(before) - len(after):5d}  {episode['title'][:52]}")
        episode["segments"] = after

    dropped = total_before - total_after
    print(f"\n  {dropped} of {total_before} segments are repetition loops "
          f"({100 * dropped / max(1, total_before):.1f}%)")
    if not args.apply:
        print("  reporting only — pass --apply to write")
        return
    # Rewrites every episode, so it replaces rather than merges — but
    # still under the lock, so it cannot interleave with a fetch.
    merge_episodes(episodes, path, replace_all=True)
    print("  written. Re-ingest: .venv/bin/python scripts/ingest_episodes.py")


if __name__ == "__main__":
    main()
