"""Find big accounts' posts the archive has something real to say under.

    .venv/bin/python scripts/find_reply_targets.py
    .venv/bin/python scripts/find_reply_targets.py --per-account 3 --hours 24
    .venv/bin/python scripts/find_reply_targets.py --only Raydium --only solana

The reply that worked: Raydium posted "I like Solana", and the archive had
Multicoin's Tushar Jain on ep 14 saying he likes Solana "for spot trading
and spot issuance" -- which is Raydium's business, in his words. Finding
that took reading Raydium's post, guessing a search, and grepping. This
does the first two for a list of accounts at once and prints the posts
where retrieval found a strong moment.

It only shortlists. Nothing here drafts or posts, and nothing it prints is
checked: the speaker labels it shows are the voiceprints, which mark only
the line they sit on and have been wrong before. Every candidate still gets
the speaker confirmed on camera and the handle confirmed through the API
before a word of it goes out.

Retrieval only -- vector search and rerank, no model answer -- so a run
costs the X reads (printed at the end) and fractions of a cent in search.

The score orders the list; it does not judge it. Calibrated on the case
this was built from: Raydium's "I like Solana" found Tushar's line at
0.60, while an airdrop tangent matched to a StonkFun post scored 0.67.
The threshold only drops what is plainly unrelated. Deciding which of the
rest is worth a reply is reading, not arithmetic.

Run on request, not on a schedule.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.podcast import PodcastIndex  # noqa: E402
from app.x_api import XCredentials  # noqa: E402

API = "https://api.x.com/2"

# Accounts in the show's orbit, with reach worth replying under. The show's
# own hosts and sponsors, the companies and people that come up on it, and
# its guests. Resolved through the API every run, so a renamed handle shows
# up as missing rather than as a wrong tag.
WATCHLIST = [
    "Raydium", "solana", "JupiterExchange", "phantom", "Polymarket",
    "HyperliquidX", "RobinhoodApp", "vladtenev", "JohannKerbrat",
    "pumpdotfun", "a1lon9", "0xMert_", "aeyakovenko", "brian_armstrong",
    "jessepollak", "LucaNetz", "ErikVoorhees", "AskVenice", "tushar_jain",
    "coinbase", "longdotxyz",
]
# Answered from the Musk archive instead. Kept apart for the same reason
# the namespaces are: a reply to Elon sourced from a Market Bubble episode
# would say the account cannot tell its archives apart.
ELON = {"elonmusk"}

# A non-owned post read, per the X pay-per-use table. Printed so the cost
# of a run is known rather than discovered on the invoice.
READ_COST = 0.005

_LINE = re.compile(r"^\[(\d[\d:]*)\]\s*(.*)$", re.M)


def credentials() -> XCredentials:
    load_dotenv(ROOT / ".env")
    return XCredentials(os.environ["X_API_KEY"], os.environ["X_API_SECRET"],
                        os.environ["X_ACCESS_TOKEN"], os.environ["X_ACCESS_SECRET"])


async def get(http, creds, url: str, params: dict) -> dict:
    response = await http.get(url, params=params,
                              headers={"Authorization": creds.header("GET", url, params)})
    response.raise_for_status()
    return response.json()


async def recent_posts(http, creds, handles: list[str], per_account: int,
                       hours: float) -> tuple[list[dict], list[str], int]:
    """Original posts (no replies, no reposts) from the last `hours`."""
    users = await get(http, creds, f"{API}/users/by",
                      {"usernames": ",".join(handles)})
    found = {u["username"].lower(): u for u in users.get("data", [])}
    missing = [h for h in handles if h.lower() not in found]
    since = (datetime.datetime.now(datetime.UTC)
             - datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    posts, reads = [], 0
    for handle in handles:
        user = found.get(handle.lower())
        if not user:
            continue
        data = await get(http, creds, f"{API}/users/{user['id']}/tweets", {
            "max_results": str(max(5, per_account)),
            "exclude": "replies,retweets",
            "start_time": since,
            "tweet.fields": "created_at,public_metrics,note_tweet",
        })
        for post in data.get("data", [])[:per_account]:
            reads += 1
            text = (post.get("note_tweet") or {}).get("text") or post["text"]
            posts.append({"handle": user["username"], "id": post["id"],
                          "text": " ".join(text.split()),
                          "likes": post["public_metrics"]["like_count"],
                          "replies": post["public_metrics"]["reply_count"],
                          "at": post["created_at"]})
    return posts, missing, reads


def best_lines(hit, query: str, n: int = 3) -> list[str]:
    """The lines of a passage that share the most words with the post."""
    want = set(re.findall(r"[a-z0-9]{4,}", query.lower()))
    lines = _LINE.findall(getattr(hit, "text_ts", "") or "")
    scored = sorted(lines, key=lambda line: -len(want & set(
        re.findall(r"[a-z0-9]{4,}", line[1].lower()))))
    return [f"[{stamp}] {said[:150]}" for stamp, said in scored[:n]]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[],
                    help="just these handles")
    ap.add_argument("--per-account", type=int, default=3)
    ap.add_argument("--hours", type=float, default=36)
    ap.add_argument("--min-score", type=float, default=0.55,
                    help="rerank score below which a moment is not shown")
    args = ap.parse_args()

    handles = args.only or WATCHLIST + sorted(ELON)
    creds = credentials()
    async with httpx.AsyncClient(timeout=30) as http:
        posts, missing, reads = await recent_posts(
            http, creds, handles, args.per_account, args.hours)

    indexes = {"podcast": PodcastIndex(), "elon": PodcastIndex(namespace="elon")}
    shortlist = []
    for post in posts:
        if len(post["text"]) < 8:
            continue                      # a bare link or an emoji
        corpus = "elon" if post["handle"].lower() in ELON else "podcast"
        try:
            hits = await indexes[corpus].retrieve(post["text"], top_k=3)
        except Exception as exc:                            # noqa: BLE001
            print(f"  search failed for @{post['handle']}: {type(exc).__name__}")
            continue
        if hits and hits[0].score >= args.min_score:
            shortlist.append((hits[0].score, post, hits[0], corpus))

    shortlist.sort(key=lambda row: -row[0])
    for score, post, hit, corpus in shortlist:
        print(f"\n  {score:.2f}  @{post['handle']}  ♥{post['likes']} "
              f"💬{post['replies']}  {post['at'][:16]}")
        print(f"        {post['text'][:200]}")
        print(f"        https://x.com/{post['handle']}/status/{post['id']}")
        print(f"     -> {corpus}: {hit.title[:70]} ({hit.published_at or '?'})"
              f" at {hit.timestamp}, voices {hit.speakers or 'unlabelled'}")
        for line in best_lines(hit, post["text"]):
            print(f"          {line}")

    print(f"\n  {len(posts)} posts read from {len(handles) - len(missing)} "
          f"accounts · {len(shortlist)} with a moment scoring "
          f">= {args.min_score} · about ${reads * READ_COST:.2f} in X reads")
    if missing:
        print(f"  not found on X: {', '.join('@' + m for m in missing)}")
    print("  Unverified: confirm the speaker on camera and the handle "
          "before anything is drafted.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
