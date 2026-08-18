"""Cut a shareable video clip from a moment in the podcast.

Search finds the moment; this turns it into something postable. The still
card the web page offers is fine attached to a link, but what people
actually watch is a short video with the words burned on.

The expensive half of a clipper is transcription, and it is already done:
the transcript was indexed with timestamps months ago, so the captions
come out of data/episodes.json and line up with the speech because they
are the same timing data.

    # find a moment, then cut it
    .venv/bin/python scripts/make_clip.py --query "why gamers make the best traders"
    .venv/bin/python scripts/make_clip.py --query "..." --pick 1

    # or set the window yourself — 44:22, 1:02:03 and raw seconds all work
    .venv/bin/python scripts/make_clip.py --episode TYX2FuacIhE --start 44:10 --end 45:05

Output is a square MP4 in clips/ — 1:1 takes the most feed height on X
without being cropped. 1080 here, because a laptop may as well produce the
best the source allows; the server-side path in app/clipper.py defaults
lower to keep egress down.

The rendering itself lives in app/clipper.py so that this script and the
service cannot drift into captioning the same episode differently.

Runs locally on purpose. Fetching and re-encoding video needs real CPU and
moves hundreds of megabytes; the free Render instance has neither, and the
residential proxy is billed per gigabyte.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.clipper import (  # noqa: E402
    build_captions,
    fetch_section,
    make_backdrop,
    render,
    stamp,
)

DATA = ROOT / "data" / "episodes.json"
OUT_DIR = ROOT / "clips"
API = "https://search.lexthedev.com"

# Full resolution locally. The service uses 720 to halve its bandwidth
# bill; nothing is being paid for per-gigabyte here.
SIZE = 1080


def load_episodes() -> dict:
    if not DATA.exists():
        sys.exit(f"{DATA} not found — run scripts/fetch_episodes.py first.")
    return {e["episode_id"]: e for e in json.loads(DATA.read_text())}


def find_moments(query: str, top_k: int = 5) -> list[dict]:
    """Ask the live search where the moment is. Semantic, so the query can
    be how you remember it rather than what was said."""
    r = httpx.post(f"{API}/v1/podcast/search",
                   json={"query": query, "top_k": top_k}, timeout=120)
    r.raise_for_status()
    return r.json().get("hits", [])


def parse_time(value: str) -> float:
    """Accept 1:02:03, 44:22 or plain seconds.

    Timestamps on the page and in search results are written the first
    way, so requiring seconds would mean doing arithmetic to use the thing
    you are looking at.
    """
    value = str(value).strip()
    if ":" not in value:
        return float(value)
    parts = [float(p) for p in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"can't read {value!r} as a time")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="find the moment by meaning")
    ap.add_argument("--pick", type=int, help="which search result to cut (1-based)")
    ap.add_argument("--episode", help="episode id, if you already know it")
    ap.add_argument("--start", help="where the clip begins, e.g. 44:22")
    ap.add_argument("--end", help="where it ends, e.g. 45:05 (overrides --duration)")
    ap.add_argument("--duration", type=float, default=50.0,
                    help="clip length; X plays up to 2:20, but 40-60s holds")
    ap.add_argument("--lead", type=float, default=3.0,
                    help="seconds before the moment, so it doesn't open mid-word "
                         "(ignored when --start is given: that is your call)")
    ap.add_argument("--size", type=int, default=SIZE, help="output square, px")
    args = ap.parse_args()

    episodes = load_episodes()

    if args.query and not args.episode:
        hits = find_moments(args.query)
        if not hits:
            sys.exit("no moments found for that query.")
        if not args.pick:
            print(f"\n  moments for {args.query!r}:\n")
            for i, h in enumerate(hits, 1):
                text = re.sub(r"\s+", " ", h["text"])[:96]
                print(f"   {i}. {h['timestamp']:>8}  {h['title'][:52]}")
                print(f"      {text}…\n")
            print("  re-run with --pick N to cut one.\n")
            return 0
        hit = hits[args.pick - 1]
        episode_id = hit["episode_id"]
        # Search suggests where the moment is; --start overrides it, because
        # a retrieval window begins where a chunk boundary fell, not where
        # the thought starts. Only the person watching knows that.
        found_at = float(hit["start_seconds"])
        start = parse_time(args.start) if args.start else max(0.0, found_at - args.lead)
    elif args.episode and args.start is not None:
        episode_id, start = args.episode, parse_time(args.start)
    else:
        sys.exit("give --query (then --pick N), or --episode with --start.")

    episode = episodes.get(episode_id)
    if not episode:
        sys.exit(f"{episode_id} is not in data/episodes.json.")

    end = parse_time(args.end) if args.end else start + args.duration
    if end <= start:
        sys.exit(f"--end ({stamp(end)}) must come after --start ({stamp(start)}).")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{episode_id}-{int(start)}.mp4"

    print(f"  episode : {episode['title'][:66]}")
    print(f"  window  : {stamp(start)} → {stamp(end)}  ({end - start:.0f}s)")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        raw = work / "raw.mp4"
        print("  fetching the section …", flush=True)
        import os
        fetch_section(episode["url"], start, end, raw,
                      os.environ.get("YTDLP_PROXY") or None)

        captions = build_captions(episode["segments"], start, end)
        print(f"  captions: {len(captions)} lines drawn from the transcript")

        backdrop = work / "backdrop.png"
        make_backdrop(episode["title"], stamp(start), backdrop, args.size)

        print("  rendering …", flush=True)
        render(raw, captions, backdrop, work, out, args.size)

    print(f"\n  {out}  ({out.stat().st_size / 1_000_000:.1f} MB)")
    print(f"  {episode['url']}&t={int(start)}s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
