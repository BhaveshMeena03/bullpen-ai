"""Drop duplicate coverage from data/episodes.json.

    .venv/bin/python scripts/dedupe_episodes.py [--dry-run]

Run after any fetch. The same broadcast reaches the index by two roads —
the YouTube upload and the X post — and where they overlap, the copy whose
citations can jump to a timestamp is the one worth keeping.

This exists as its own step, rather than living inside one fetcher, because
the duplicate can arrive from either direction: an X broadcast indexed on
Monday and the YouTube upload landing on Friday by cron is just as much a
duplicate as the reverse, and a guard inside the X fetcher never sees it.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.dedupe import SAME_RECORDING, dedupe, describe  # noqa: E402

OUT = ROOT / "data" / "episodes.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(OUT))
    ap.add_argument("--threshold", type=float, default=SAME_RECORDING)
    ap.add_argument("--apply", action="store_true",
                    help="actually remove them; without this it only reports")
    args = ap.parse_args()

    path = Path(args.file)
    episodes = json.loads(path.read_text())
    kept, dropped = dedupe(episodes, threshold=args.threshold)

    print(f"  {len(episodes)} episode(s) in {path.name}")
    print("  " + describe(dropped))

    if not args.apply:
        # Reporting is the default because the destructive version of this
        # was wrong once already, and a wrong deletion here is silent: the
        # service keeps answering, slightly worse, with no error anywhere.
        print("\n  reporting only — pass --apply to remove them")
        return
    if not dropped:
        return
    path.write_text(json.dumps(kept, ensure_ascii=False))
    print(f"\n  {len(kept)} episode(s) kept -> {path.name}")
    print("  Re-ingest so the index matches: "
          ".venv/bin/python scripts/ingest_episodes.py")


if __name__ == "__main__":
    main()
