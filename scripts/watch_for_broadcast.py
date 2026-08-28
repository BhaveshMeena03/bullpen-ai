"""Wait for the show to finish, then index it without being asked.

    .venv/bin/python scripts/watch_for_broadcast.py
    .venv/bin/python scripts/watch_for_broadcast.py --dry-run
    scripts/watch_for_broadcast.py --install     # via the shell wrapper

The archive's value is being current: somebody asks about last night's
show while people are still talking about it, or they ask a day late and
the answer is that it is not indexed yet. The gap between the two was
whoever remembered to run add_broadcast.

Nine broadcasts, nine Thursdays, every one posted between 20:31 and 20:39
UTC — Friday 02:01-02:09 IST — running three to four hours. So the
recording is ready somewhere around 05:00-06:30 IST, which is a bad time
to depend on a person.

Knowing WHEN it is ready is the whole problem. The post exists from the
first minute of the stream, so its presence proves nothing. What changes
is the video's reported duration: it grows while the show is live and
stops when X swaps the live feed for the finished recording. Two equal
readings, fifteen minutes apart, past a plausible length, is the end of
the show — and that is a fact about the video rather than a guess about
the clock.

It indexes. It does not post. The summary it produces goes to the site
and to anyone who asks, and nothing goes out under the account's name
unprompted, because a summary nobody has read is not something to publish
while its author is asleep.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.episode_store import load  # noqa: E402
from app.x_api import API, XCredentials  # noqa: E402
from scripts.find_new_broadcasts import (  # noqa: E402
    LOOKS_LIKE_AN_EPISODE,
    SHOW,
)

EPISODES = ROOT / "data" / "episodes.json"


# What one reading of the video's length means, given the reading before
# it. Separated from the loop because this decision starts an unattended
# twenty minute job on somebody's laptop at six in the morning, and the
# only way to know it is right is to be able to run it without waiting
# for a Thursday.
WAIT_SHORT = "short"       # no finished show has ever been this brief
WAIT_FIRST = "first"       # nothing to compare against yet
WAIT_GROWING = "growing"   # the number went up, so the stream is live
READY = "ready"            # past a plausible length and no longer moving


def verdict(duration_ms: int, previous: int | None,
            min_hours: float = 2.5) -> str:
    if duration_ms < min_hours * 3_600_000:
        return WAIT_SHORT
    if previous is None:
        return WAIT_FIRST
    if duration_ms > previous:
        return WAIT_GROWING
    return READY


def say(message: str) -> None:
    print(f"  {datetime.now(UTC):%H:%M} UTC  {message}", flush=True)


def notify(title: str, body: str) -> None:
    subprocess.run(["osascript", "-e",
                    f'display notification "{body}" with title "{title}"'],
                   capture_output=True)


async def candidate(http: httpx.AsyncClient, cred: XCredentials,
                    user_id: str) -> tuple[str, int] | None:
    """The newest un-indexed broadcast post, and its reported length."""
    url = f"{API}/users/{user_id}/tweets"
    params = {"max_results": "10",
              "tweet.fields": "created_at,attachments,referenced_tweets",
              "expansions": "attachments.media_keys",
              "media.fields": "type,duration_ms"}
    response = await http.get(
        url, params=params,
        headers={"Authorization": cred.header("GET", url, params)})
    if response.status_code != 200:
        say(f"could not read @{SHOW}: HTTP {response.status_code}")
        return None

    payload = response.json()
    media = {m["media_key"]: m
             for m in (payload.get("includes", {}).get("media") or [])}
    indexed = {e["episode_id"] for e in load(EPISODES)}

    for post in (payload.get("data") or []):
        if any(r.get("type") in ("replied_to", "quoted")
               for r in (post.get("referenced_tweets") or [])):
            continue
        if f"x-{post['id']}" in indexed:
            continue
        keys = (post.get("attachments") or {}).get("media_keys") or []
        if not any(media.get(k, {}).get("type") == "video" for k in keys):
            continue
        if not LOOKS_LIKE_AN_EPISODE.search(post.get("text", "")):
            continue
        longest = max((media.get(k, {}).get("duration_ms") or 0
                       for k in keys), default=0)
        return post["id"], longest
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=int, default=900,
                    help="how often to look. Fifteen minutes is also the "
                         "gap that has to show no growth before the stream "
                         "counts as ended.")
    ap.add_argument("--window-hours", type=float, default=9.0,
                    help="give up after this long and let a person handle it")
    ap.add_argument("--min-hours", type=float, default=2.5,
                    help="a finished show has never been shorter")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what it would do; never starts the pipeline")
    args = ap.parse_args()

    settings = get_settings()
    cred = XCredentials(settings.x_api_key, settings.x_api_secret,
                        settings.x_access_token, settings.x_access_secret)
    deadline = time.time() + args.window_hours * 3600
    seen: dict[str, int] = {}
    polls = 0

    say(f"watching @{SHOW} for {args.window_hours:g}h, "
        f"every {args.poll_seconds // 60}min")

    async with httpx.AsyncClient(timeout=30) as http:
        url = f"{API}/users/by/username/{SHOW}"
        response = await http.get(
            url, headers={"Authorization": cred.header("GET", url)})
        if response.status_code != 200:
            say(f"could not resolve @{SHOW}: HTTP {response.status_code}")
            return 2
        user_id = (response.json().get("data") or {}).get("id")

        while time.time() < deadline:
            polls += 1
            found = await candidate(http, cred, user_id)
            if not found:
                say("nothing un-indexed yet")
            else:
                post_id, duration = found
                hours = duration / 3_600_000
                before = seen.get(post_id)
                seen[post_id] = duration

                state = verdict(duration, before, args.min_hours)
                if state == WAIT_SHORT:
                    say(f"{post_id}: {hours:.1f}h — still short, waiting")
                elif state == WAIT_FIRST:
                    say(f"{post_id}: {hours:.1f}h — need one more reading "
                        f"to know it has stopped growing")
                elif state == WAIT_GROWING:
                    say(f"{post_id}: {hours:.1f}h — still growing, so the "
                        f"stream is live")
                else:
                    # Two equal readings past a plausible length: the video
                    # X is serving is the finished recording, not the feed.
                    say(f"{post_id}: {hours:.1f}h and unchanged — the show "
                        f"has ended")
                    link = f"https://x.com/{SHOW}/status/{post_id}"
                    if args.dry_run:
                        say(f"--dry-run, so stopping here. Would run: {link}")
                        return 0
                    notify("Market Bubble",
                           "Show ended — indexing now, about 20 minutes.")
                    # caffeinate: the whole point is that this runs while
                    # nobody is at the machine, and a Mac that sleeps
                    # mid-transcription leaves a half-written episode.
                    done = subprocess.run(
                        ["caffeinate", "-i", sys.executable,
                         "scripts/add_broadcast.py", link], cwd=ROOT)
                    if done.returncode == 0:
                        notify("Market Bubble", "Indexed and searchable.")
                        say("indexed")
                        return 0
                    notify("Market Bubble",
                           "Indexing FAILED — see the log.")
                    say(f"add_broadcast exited {done.returncode}")
                    return 1

            await asyncio.sleep(args.poll_seconds)

    say(f"window closed after {polls} polls — nothing was ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
