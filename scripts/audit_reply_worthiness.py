"""Which replies should never have gone out.

    .venv/bin/python scripts/audit_reply_worthiness.py --limit 100

audit_posted_replies.py asks whether a reply was TRUE. This asks whether
it should have existed. Those fail differently: a perfectly accurate
citation posted under a scam thread still puts this account under a scam
thread, and no amount of checking the timestamp catches it.

For every reply on the timeline it fetches the post it answered and runs
the guards the bot uses today over that parent. Three outcomes:

  covered   a guard now catches it — the reply predates the guard, so
            nothing to do but note it
  MISSED    every guard passes and the reply still should not have gone
            out — this is the list worth reading
  fine      a real question, answered

Only the parent is judged, because that is all the bot sees when it
decides. Reading our own reply to decide whether to have written it is
hindsight the live code does not get.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.x_api import API, XCredentials  # noqa: E402
from app import x_bot  # noqa: E402

# Each guard, named the way it would be reported. Order matches the order
# the mention loop applies them, so the first hit is the one that would
# actually have stopped the reply.
GUARDS = (
    ("bait / scam / solicitation", lambda t: x_bot.looks_like_bait(t)),
    ("addressed to another assistant", x_bot.addressed_to_another_bot),
    ("asked to be left alone", x_bot.asks_to_be_left_alone),
    ("about the token, not the show", x_bot.asks_about_us),
)


async def get(http, cred, path, params):
    url = f"{API}{path}"
    r = await http.get(url, params=params,
                       headers={"Authorization": cred.header("GET", url, params)})
    return r


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the full parent/reply pairs here")
    args = ap.parse_args()

    s = get_settings()
    cred = XCredentials(s.x_api_key, s.x_api_secret,
                        s.x_access_token, s.x_access_secret)

    async with httpx.AsyncClient(timeout=30) as http:
        r = await get(http, cred, f"/users/{s.x_bot_user_id}/tweets", {
            "max_results": str(max(5, min(args.limit, 100))),
            "tweet.fields": "created_at,note_tweet,referenced_tweets,"
                            "public_metrics",
        })
        if r.status_code != 200:
            print(f"  could not read the timeline: HTTP {r.status_code}")
            return 2
        posts = r.json().get("data") or []

        # The parents, in one call per hundred rather than one per reply.
        parents = {}
        for p in posts:
            for ref in p.get("referenced_tweets") or []:
                if ref["type"] == "replied_to":
                    parents[ref["id"]] = None
        ids = list(parents)
        for i in range(0, len(ids), 100):
            chunk = ids[i:i + 100]
            r = await get(http, cred, "/tweets", {
                "ids": ",".join(chunk),
                "tweet.fields": "created_at,note_tweet,author_id",
                "expansions": "author_id",
                "user.fields": "username,verified,public_metrics",
            })
            if r.status_code != 200:
                print(f"  could not read parents: HTTP {r.status_code}")
                continue
            body = r.json()
            users = {u["id"]: u for u in
                     (body.get("includes", {}).get("users") or [])}
            for t in body.get("data") or []:
                t["author"] = users.get(t.get("author_id"), {})
                parents[t["id"]] = t

    rows, missed, covered, replies = [], 0, 0, 0
    for p in posts:
        parent_id = next((ref["id"] for ref in (p.get("referenced_tweets") or [])
                          if ref["type"] == "replied_to"), None)
        if not parent_id:
            continue                            # a timeline post, not a reply
        replies += 1
        parent = parents.get(parent_id)
        if not parent:
            rows.append({"reply_id": p["id"], "verdict": "parent unavailable"})
            continue

        text = (parent.get("note_tweet") or {}).get("text") or parent["text"]
        question = x_bot.question_from(text)
        hit = next((name for name, fn in GUARDS if fn(question or text)), None)
        body = (p.get("note_tweet") or {}).get("text") or p["text"]
        row = {
            "reply_id": p["id"],
            "when": p["created_at"][:16],
            "from": "@" + (parent.get("author", {}).get("username") or "?"),
            "parent": text,
            "reply": body,
            "guard": hit,
            "is_question": x_bot.asks_something(question or text),
            "social": x_bot.reads_as_social(question or text),
        }
        if hit:
            covered += 1
            row["verdict"] = "covered"
        elif not row["is_question"]:
            missed += 1
            row["verdict"] = "MISSED"
        else:
            row["verdict"] = "fine"
        rows.append(row)

    for row in rows:
        if row.get("verdict") not in ("MISSED", "covered"):
            continue
        print(f"  {'=' * 70}")
        print(f"  {row['verdict']}  {row.get('when','')}  {row.get('from','')}"
              f"  https://x.com/mbubbleSearch/status/{row['reply_id']}")
        if row.get("guard"):
            print(f"      guard now: {row['guard']}")
        for line in (row.get("parent") or "").splitlines()[:4]:
            print(f"    them | {line[:78]}")
        for line in (row.get("reply") or "").splitlines()[:3]:
            print(f"      us | {line[:78]}")

    print(f"\n  {replies} replies · {covered} already covered by a guard · "
          f"{missed} still missed\n")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
        print(f"  full pairs -> {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
