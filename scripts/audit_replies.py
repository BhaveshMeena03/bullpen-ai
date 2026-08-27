"""Check what the bot actually posted, not what it would post.

    .venv/bin/python scripts/audit_replies.py
    .venv/bin/python scripts/audit_replies.py --limit 40
    .venv/bin/python scripts/audit_replies.py --since 2026-08-27T05:00:00Z

Everything else in this repo tests the composer: given a mention, what
would the bot say. That is most of the value and it is not the whole of
it, because the composer runs here and the replies happen there. Today a
build failed and the previous image kept serving; the composer was fine
and the account was broken.

So this reads the account's own timeline and runs the checks the sweeps
run — markup, stray asterisks, retrieval plumbing, walls of text, deleted
project names, replies cut mid-sentence, tagged bystanders — over posts
that really went out.

Costs $0.001 per post read, so a full pass over the last fifty is five
cents. Reads only; it never posts, deletes or edits anything.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.x_api import API, XCredentials  # noqa: E402
from app.x_bot import weighted_length  # noqa: E402

# Hosts a reply is allowed to link to. Anything else is either a
# hallucination or something read out of a transcript.
OURS = ("youtube.com", "youtu.be", "x.com/MarketBubble", "t.co",
        "lexthedev.com")

# Project names that look like domains. These were being deleted from the
# middle of sentences, so their absence is worth noticing, but only the
# broken shape can be detected automatically: a name followed by a capital
# where the rest of the sentence should be.
_EATEN_NAME = re.compile(r"\b(?:and|with|between|on|to|via)\s+[A-Z][a-z]+\s+"
                         r"(?:The|This|That|It)\b")


def defects(text: str, limit: int) -> list[str]:
    found = []
    low = text.lower()

    if re.search(r"</?[a-z_]+>", text):
        found.append("MARKUP")
    if re.search(r"(?<!\*)\*(?!\s)[^*\n]+(?<!\s)\*(?!\*)", text):
        found.append("ASTERISKS")
    if "excerpt" in low or "unnamed speaker" in low or "identity unclear" in low:
        found.append("PLUMBING")
    if "i've" in text or "i couldn't" in text:
        found.append("LOWERCASE-I")

    body = re.split(r"\n\n(?:Jump to|Full episode)", text)[0]
    is_summary = "\nTopics\n" in text or "\nNotable moments\n" in text
    if not is_summary and len(body) > 340 and "\n\n" not in body:
        found.append("WALL-OF-TEXT")
    if weighted_length(text) > (4000 if is_summary else limit):
        found.append("OVER-LIMIT")

    # The leading @handle of a reply is X's own, not something the bot
    # wrote, so only handles after the first line count.
    after_lead = re.sub(r"^(?:@\w{1,15}\s+)+", "", text)
    if re.search(r"@\w{2,15}", after_lead):
        found.append("TAGS-SOMEONE")

    if re.search(r"(?i)\b(?:as an ai|i am (?:an? )?(?:ai|language model)|"
                 r"my (?:system )?prompt|my instructions)\b", text):
        found.append("IDENTITY-LEAK")
    for url in re.findall(r"https?://[^\s]+", text):
        if not any(host in url for host in OURS):
            found.append("FOREIGN-URL")
    if _EATEN_NAME.search(body):
        found.append("MAYBE-EATEN-NAME")
    if body.rstrip().endswith(("the", "a", "and", "to", "of", "in", "-")):
        found.append("CUT-MID-SENTENCE")
    # A bare miss is not a defect, but it is worth counting: a run of them
    # means retrieval or the archive, not formatting.
    if re.match(r"(?i)^(?:@\w+\s+)*i (?:couldn't|looked)", text):
        found.append("bare-miss")
    return found


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25,
                    help="how many recent posts to read (max 100)")
    ap.add_argument("--since", help="only posts at or after this ISO time")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the posts with defects")
    args = ap.parse_args()

    settings = get_settings()
    cred = XCredentials(settings.x_api_key, settings.x_api_secret,
                        settings.x_access_token, settings.x_access_secret)
    url = f"{API}/users/{settings.x_bot_user_id}/tweets"
    params = {"max_results": str(max(5, min(args.limit, 100))),
              "tweet.fields": "created_at,note_tweet,referenced_tweets"}

    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.get(
            url, params=params,
            headers={"Authorization": cred.header("GET", url, params)})
    if response.status_code != 200:
        print(f"could not read the timeline: HTTP {response.status_code}")
        print(response.text[:300])
        return 2

    posts = response.json().get("data") or []
    # Replies only. The same account also posts by hand — the announcement
    # thread tags @mbubbleSearch and @MarketBubble on purpose — and auditing
    # a human's own post as though the bot wrote it reports a defect that is
    # not one.
    written_by_hand = [
        p for p in posts
        if not any(r.get("type") == "replied_to"
                   for r in (p.get("referenced_tweets") or []))]
    posts = [p for p in posts if p not in written_by_hand]
    if args.since:
        posts = [p for p in posts if p.get("created_at", "") >= args.since]

    clean, flagged, misses = 0, [], 0
    for post in posts:
        # note_tweet carries the whole body; `text` is a 280-char preview,
        # and auditing the preview would report every long reply as cut off.
        body = (post.get("note_tweet") or {}).get("text") or post["text"]
        problems = [d for d in defects(body, settings.x_bot_post_limit)
                    if d != "bare-miss"]
        if "bare-miss" in defects(body, settings.x_bot_post_limit):
            misses += 1
        if problems:
            flagged.append((post, body, problems))
        else:
            clean += 1
            if not args.quiet:
                first = body.splitlines()[0] if body.splitlines() else ""
                print(f"  ok   {post['created_at']}  {first[:76]}")

    for post, body, problems in flagged:
        print("\n" + "=" * 72)
        print(f"  !!! {' '.join(problems)}   {post['created_at']}")
        print(f"      https://x.com/i/status/{post['id']}")
        print("=" * 72)
        for line in body.splitlines():
            print(f"  | {line}")

    read_cost = (len(posts) + len(written_by_hand)) * 0.001
    print(f"\n  {len(posts)} replies audited · {clean} clean · "
          f"{len(flagged)} flagged · {misses} honest misses · "
          f"${read_cost:.3f}")
    if written_by_hand:
        print(f"  ({len(written_by_hand)} post(s) written by hand, skipped)")
    if flagged:
        print("  A flagged reply is still live — delete it from the account "
              "if it is bad enough to matter.")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
