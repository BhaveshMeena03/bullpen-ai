"""Find Market Bubble live broadcasts that are not indexed yet.

    .venv/bin/python scripts/find_new_broadcasts.py
    .venv/bin/python scripts/find_new_broadcasts.py --since 2026-08-01
    .venv/bin/python scripts/find_new_broadcasts.py --quiet   # exit code only

The live broadcasts are the half of the archive nobody else has — roughly a
third of every show never reaches the YouTube upload. They were also the
half that only got indexed when somebody remembered, because fetch_x_episodes
takes URLs by hand: yt-dlp cannot enumerate an account's videos, so there was
no way to ask "what is new".

There is now. The bot already holds X API credentials, so the account's own
timeline answers the question directly. This finds broadcast posts from
@MarketBubble, compares them against what is already in data/episodes.json,
and prints the ones missing along with the command that indexes them.

It does not index anything. Transcription runs locally on Apple Silicon and
takes about twenty minutes for a four-hour show, which is not something to
start unattended from a cron line — and the pipeline it feeds asks for a
human at the end anyway, because a summary goes out under the account's name.

Costs one user read per post examined, about $0.10 a check.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.episode_store import load  # noqa: E402
from app.x_api import API, XCredentials  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
SHOW = "MarketBubble"

# A broadcast post carries a video and reads like an episode title. Replies
# and plain commentary do not, so both signals are required — one alone
# picks up every clip and screenshot the account posts between shows.
LOOKS_LIKE_AN_EPISODE = re.compile(
    r"""(?ix) \b(?: live\s+w/ | market\s+bubble | ep(?:isode)?\s*\#?\s*\d
                  | presented\s+by | draft\s+night )\b""")


def new_broadcasts(posts: list[dict], media: dict[str, dict],
                   indexed: set[str], min_hours: float = 2.0,
                   since: str | None = None, settle_hours: float = 6.0,
                   now: datetime | None = None) -> tuple[list, int]:
    """Which of these posts are full broadcasts nobody has indexed yet.

    Returns the ones ready to index and a count of the ones that are too
    recent to touch, which are reported rather than hidden.

    Pure, so the filtering can be tested without spending an API call on
    every run of the suite.
    """
    now = now or datetime.now(UTC)
    ready_after = now - timedelta(hours=settle_hours)
    found, too_soon = [], 0
    for post in posts:
        # A reply is a comment on something, not a broadcast of its own.
        if any(r.get("type") in ("replied_to", "quoted")
               for r in (post.get("referenced_tweets") or [])):
            continue
        keys = (post.get("attachments") or {}).get("media_keys") or []
        has_video = any(media.get(k, {}).get("type") in ("video", "animated_gif")
                        for k in keys)
        if not has_video or not LOOKS_LIKE_AN_EPISODE.search(post.get("text", "")):
            continue
        if since and post.get("created_at", "")[:10] < since:
            continue
        if f"x-{post['id']}" in indexed:
            continue
        longest = max((media.get(k, {}).get("duration_ms") or 0 for k in keys),
                      default=0)
        # A known-short video is one of the 30-60 minute cut-downs the
        # account posts between shows; its content is already inside the
        # full broadcast, so indexing it would duplicate passages and split
        # citations for the same moment across two sources.
        #
        # An unknown length is kept and flagged rather than dropped: X does
        # not always report duration on a live post, and silently skipping
        # a whole show because a field was missing is the worse failure.
        if longest and longest < min_hours * 3_600_000:
            continue

        # The show goes live for three or four hours and the post exists
        # from the first minute of it. Fetched mid-stream you get whatever
        # has aired so far, and the archive would hold a truncated episode
        # that nothing later re-checks. So a post is only ready once it is
        # old enough that the broadcast has ended and X has finished
        # replacing the live stream with the full recording.
        try:
            posted = datetime.fromisoformat(
                post["created_at"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            posted = ready_after  # undated: treat as ready rather than lost
        if posted > ready_after:
            too_soon += 1
            continue
        found.append((post, longest))
    return found, too_soon


async def show_id(http: httpx.AsyncClient, cred: XCredentials) -> str | None:
    url = f"{API}/users/by/username/{SHOW}"
    response = await http.get(
        url, headers={"Authorization": cred.header("GET", url)})
    if response.status_code != 200:
        print(f"  could not resolve @{SHOW}: HTTP {response.status_code}")
        return None
    return (response.json().get("data") or {}).get("id")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20,
                    help="how many recent posts to examine")
    ap.add_argument("--since", metavar="YYYY-MM-DD",
                    help="ignore anything older")
    ap.add_argument("--min-hours", type=float, default=2.0,
                    help="skip anything shorter. The account posts 30-60 "
                         "minute cut-downs between shows and those are "
                         "already inside the full broadcast — indexing them "
                         "would duplicate passages and split citations "
                         "across two sources for the same moment.")
    ap.add_argument("--settle-hours", type=float, default=6.0,
                    help="ignore posts newer than this. The broadcast post "
                         "goes up when the stream starts, not when it ends, "
                         "so fetching too early captures a partial episode.")
    ap.add_argument("--quiet", action="store_true",
                    help="print nothing; exit 1 if something is missing")
    args = ap.parse_args()

    settings = get_settings()
    cred = XCredentials(settings.x_api_key, settings.x_api_secret,
                        settings.x_access_token, settings.x_access_secret)

    # Already indexed, by the id the fetcher assigns: "x-<post id>".
    indexed = {e["episode_id"] for e in load(EPISODES)}

    async with httpx.AsyncClient(timeout=30) as http:
        user_id = await show_id(http, cred)
        if not user_id:
            return 2
        url = f"{API}/users/{user_id}/tweets"
        params = {
            "max_results": str(max(5, min(args.limit, 100))),
            "tweet.fields": "created_at,attachments,referenced_tweets",
            "expansions": "attachments.media_keys",
            "media.fields": "type,duration_ms",
        }
        response = await http.get(
            url, params=params,
            headers={"Authorization": cred.header("GET", url, params)})

    if response.status_code != 200:
        print(f"  could not read @{SHOW}: HTTP {response.status_code}")
        print(response.text[:200])
        return 2

    payload = response.json()
    media = {m["media_key"]: m
             for m in (payload.get("includes", {}).get("media") or [])}

    missing, too_soon = new_broadcasts(
        payload.get("data") or [], media, indexed,
        min_hours=args.min_hours, since=args.since,
        settle_hours=args.settle_hours)

    if args.quiet:
        return 1 if missing else 0

    def waiting() -> None:
        if too_soon:
            print(f"\n  {too_soon} broadcast(s) posted in the last "
                  f"{args.settle_hours:g}h — still live, or X is still "
                  f"processing the recording. Check again later.")

    if not missing:
        print(f"  nothing new — every broadcast @{SHOW} has posted in the "
              f"last {args.limit} posts is already indexed")
        waiting()
        return 0

    print(f"  {len(missing)} broadcast(s) not indexed:\n")
    for post, duration in missing:
        hours = duration / 3_600_000 if duration else 0
        # Minutes under the hour: a one-minute teaser printed as "0.0h"
        # reads like a missing value rather than a short clip.
        if not duration:
            length = "length unknown  "
        elif hours < 1:
            length = f"{duration / 60_000:.0f}m  "
        else:
            length = f"{hours:.1f}h  "
        print(f"  {post['created_at'][:10]}  {length}{post['text'][:64]}")
        print("    .venv/bin/python scripts/add_broadcast.py \\")
        print(f"      https://x.com/{SHOW}/status/{post['id']}\n")
    waiting()
    print(f"\n  Clips under {args.min_hours:g}h are skipped — the account posts "
          f"cut-downs between shows and they are already inside the full "
          f"broadcast.")
    print("  Transcription runs locally and takes about twenty minutes for a "
          "four-hour show.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
