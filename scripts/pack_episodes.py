#!/usr/bin/env python3
"""Compress data/episodes.json into the copy the image ships.

Run this after an ingest. episodes.json is 7.3MB and rewritten every time
new episodes land, so committing it would put a fresh 7MB blob in git on
each pass; it stays gitignored and this 2.3MB gzip is what gets committed
and copied into the container.

The clipper is the only thing that reads it at runtime — it needs the
source URL to download a section and the per-second segments to build
captions, and neither exists anywhere else at runtime. Forgetting to run
this means newly ingested episodes return 404 from the clip endpoint;
nothing else degrades.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "episodes.json"
OUT = ROOT / "data" / "episodes.json.gz"


def main() -> int:
    if not RAW.exists():
        print(f"no {RAW} — nothing to pack", file=sys.stderr)
        return 1

    # Parsed rather than streamed straight through, so a truncated or
    # half-written episodes.json fails here instead of shipping and
    # failing in the container.
    episodes = json.loads(RAW.read_text())
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(episodes, fh, separators=(",", ":"))

    before = RAW.stat().st_size / 1e6
    after = OUT.stat().st_size / 1e6
    print(f"  {len(episodes)} episodes")
    print(f"  {before:.1f}MB -> {after:.1f}MB  ({OUT.relative_to(ROOT)})")
    print("  commit it: git add -f data/episodes.json.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
