"""Cut a captioned clip from any indexed episode, on this machine.

    .venv/bin/python scripts/make_clip.py "what did luca netz say about pudgy penguins"
    .venv/bin/python scripts/make_clip.py --episode x-2075316750439338088 --at 4:01:47
    .venv/bin/python scripts/make_clip.py "..." --seconds 60 --height 720

Local rather than a web feature, deliberately. The hosted service runs on a
free tier with 0.1 CPU: ffmpeg encodes at about 2.6x realtime on this laptop
at full CPU, so the same work there would take minutes per clip AND block
every search request while it ran. Trading a search engine that works for a
clip button that might is the wrong trade. This gets the actual value — a
shareable clip of something nobody else has — at no hosting risk, and it
proves whether clips are worth hosting before anything is paid for.

Works for both sources. YouTube clips are a convenience, since a viewer
could scrub there themselves. A clip from an X broadcast is the only way
anyone gets that moment: about half of every live show never reaches the
upload, and X cannot deep-link to a timestamp at all.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.clipper import (  # noqa: E402
    build_captions,
    fetch_section,
    ffmpeg_available,
    make_backdrop,
    render,
    stamp,
)

EPISODES = ROOT / "data" / "episodes.json"
SEARCH = "https://search.lexthedev.com"


def parse_timestamp(value: str) -> float:
    """"4:01:47", "56:47" or "3407" -> seconds."""
    parts = [p for p in str(value).strip().split(":") if p != ""]
    if not parts:
        raise ValueError(f"could not read a timestamp from {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def top_hit(query: str) -> dict:
    """Ask the live search where the best moment for this question is."""
    request = urllib.request.Request(
        f"{SEARCH}/v1/podcast/search",
        data=json.dumps({"query": query}).encode(),
        headers={"content-type": "application/json"})
    body = json.loads(urllib.request.urlopen(request, timeout=180).read(),
                      strict=False)
    hits = body.get("hits") or []
    if not hits:
        sys.exit(f"no results for {query!r}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?",
                    help="a question; the top result becomes the clip")
    ap.add_argument("--episode", help="episode id, instead of a query")
    ap.add_argument("--at", help="timestamp, e.g. 4:01:47 (with --episode)")
    ap.add_argument("--seconds", type=float, default=45.0,
                    help="clip length (default 45)")
    ap.add_argument("--lead", type=float, default=3.0,
                    help="seconds of run-up before the moment, so it does "
                         "not open mid-word (default 3)")
    ap.add_argument("--height", type=int, default=1080,
                    choices=[480, 720, 1080], help="output height")
    ap.add_argument("--out", help="output file (default: ~/Desktop)")
    args = ap.parse_args()

    if not ffmpeg_available():
        sys.exit("ffmpeg is not on PATH — brew install ffmpeg")

    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}

    if args.query:
        hit = top_hit(args.query)
        episode_id = hit["episode_id"]
        start = float(hit["start_seconds"])
        print(f"  top result: {hit['title'][:56]} @ {hit['timestamp']}")
    else:
        if not (args.episode and args.at):
            sys.exit("give a query, or --episode with --at")
        episode_id, start = args.episode, parse_timestamp(args.at)

    episode = episodes.get(episode_id)
    if episode is None:
        sys.exit(f"{episode_id} is not in {EPISODES.name} — re-fetch it first")

    # A clip that opens mid-syllable reads as broken, so back up a little.
    start = max(0.0, start - args.lead)
    end = start + args.seconds
    on_x = episode.get("platform") != "youtube"

    print(f"  {episode['title'][:60]}")
    print(f"  {stamp(start)} → {stamp(end)}  ({args.seconds:.0f}s, "
          f"{args.height}p, {'X broadcast' if on_x else 'YouTube'})")

    captions = build_captions(episode["segments"], start, end)
    out = Path(args.out) if args.out else (
        Path.home() / "Desktop" /
        f"clip-{episode_id}-{int(start)}s-{args.height}p.mp4")

    began = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        source = work / "src.mp4"
        print("  downloading the section…")
        fetch_section(episode["url"], start, end, source, height=args.height)

        backdrop = work / "backdrop.png"
        make_backdrop(episode["title"], stamp(start), backdrop, args.height)

        print(f"  rendering {len(captions)} caption(s)…")
        render(source, captions, backdrop, work, out, args.height)

    size_mb = out.stat().st_size / 1_048_576
    print(f"\n  {out}")
    print(f"  {size_mb:.1f} MB in {time.time() - began:.0f}s")
    if on_x:
        # Worth saying out loud: this is the half of the show that the
        # YouTube upload cuts, and X cannot link to a timestamp, so the clip
        # is the only way to point anyone at this moment.
        print("  (from the live broadcast — this moment is not on YouTube, "
              "and X cannot link to a timestamp)")
    if shutil.which("open"):
        print("  open it:  open " + str(out).replace(" ", "\\ "))


if __name__ == "__main__":
    main()
