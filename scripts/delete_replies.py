"""Delete replies this account posted, by conversation or by id.

    .venv/bin/python scripts/delete_replies.py --thread 2093...      # list
    .venv/bin/python scripts/delete_replies.py --thread 2093... --keep 1 --yes
    .venv/bin/python scripts/delete_replies.py --id 2093... --yes

Written for a loop: @clawpumptech is an automated account too, its reply
mentioned this one, that reply mentioned it back, and neither stopped. The
cap in x_bot.py prevents the next one; this clears up the last one.

Nothing is deleted without --yes. Without it every candidate is printed in
full, because a deletion cannot be taken back and a reply somebody has
already read is not really gone.

--keep N leaves the first N replies in the thread alone, which is usually
what you want: the first answer was a real one, and everything after it is
the loop.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.x_api import API, XCredentials  # noqa: E402


def _id(value: str) -> str:
    value = value.strip().rstrip("/")
    return value.rsplit("/", 1)[-1].split("?")[0]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", help="conversation id or any post URL in it")
    ap.add_argument("--id", action="append", default=[],
                    help="delete one specific post; repeatable")
    ap.add_argument("--keep", type=int, default=1,
                    help="leave the first N replies in the thread (default 1)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--yes", action="store_true",
                    help="actually delete; without it, nothing is removed")
    args = ap.parse_args()

    if not args.thread and not args.id:
        ap.error("give --thread or --id")

    settings = get_settings()
    cred = XCredentials(settings.x_api_key, settings.x_api_secret,
                        settings.x_access_token, settings.x_access_secret)

    targets: list[tuple[str, str, str]] = []
    if args.id:
        targets = [(_id(x), "", "(named directly)") for x in args.id]
    else:
        url = f"{API}/users/{settings.x_bot_user_id}/tweets"
        params = {"max_results": str(max(5, min(args.limit, 100))),
                  "tweet.fields": "created_at,conversation_id,note_tweet"}
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.get(
                url, params=params,
                headers={"Authorization": cred.header("GET", url, params)})
        if response.status_code != 200:
            print(f"could not read the timeline: HTTP {response.status_code}")
            return 2
        thread = _id(args.thread)
        mine = [p for p in (response.json().get("data") or [])
                if p.get("conversation_id") == thread]
        mine.sort(key=lambda p: p["created_at"])          # oldest first
        for post in mine[args.keep:]:
            body = (post.get("note_tweet") or {}).get("text") or post["text"]
            targets.append((post["id"], post["created_at"], body))
        print(f"  {len(mine)} replies by this account in that thread, "
              f"keeping the first {args.keep}\n")

    if not targets:
        print("  nothing to delete\n")
        return 0

    for post_id, when, body in targets:
        print("=" * 70)
        print(f"  {when}  https://x.com/mbubbleSearch/status/{post_id}")
        for line in body.splitlines()[:4]:
            print(f"  | {line[:78]}")

    if not args.yes:
        print(f"\n  {len(targets)} would be deleted. Nothing was removed. "
              f"Add --yes to do it.\n")
        return 0

    gone = 0
    for post_id, _, _ in targets:
        url = f"{API}/tweets/{post_id}"
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.request(
                "DELETE", url,
                headers={"Authorization": cred.header("DELETE", url)})
        if response.status_code == 200:
            gone += 1
            print(f"  deleted {post_id}")
        else:
            print(f"  could NOT delete {post_id}: HTTP "
                  f"{response.status_code} {response.text[:120]}")
    print(f"\n  {gone} of {len(targets)} deleted\n")
    return 0 if gone == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
