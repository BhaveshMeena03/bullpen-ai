"""Find the player URL for every X broadcast in the archive.

    .venv/bin/python scripts/fetch_broadcast_links.py
    .venv/bin/python scripts/fetch_broadcast_links.py --write

Why this exists, and it is worth writing down because the reasoning went
wrong twice before landing here.

A citation into a broadcast used to point at the STATUS url:

    https://x.com/MarketBubble/status/2088002434253922358?t=7774

Loaded directly in a browser that works — the player opens at 7774,
checked on three broadcasts. What nobody checked is what X does with that
url inside a POST. It renders it as an embedded quote card, and a card is
not a link with a query string: clicking it opens the quoted post at the
top. So every reply said "Jump to 2:09:34" above a card that could not
jump anywhere, and the timestamp was decoration.

The broadcast player has its own url, which the show's own post links to:

    https://x.com/i/broadcasts/1kJzDPPQlqyKv?t=7774

That is not a status, so it does not become a quote card, and it seeks.
This fetches that url for each broadcast and writes the mapping.

Kept as a side file rather than folded into episodes.json on purpose. The
url a citation is built from lives in Pinecone metadata, written at ingest
time; changing episodes.json would fix nothing without re-embedding the
whole archive. A mapping consulted when the link is BUILT costs one file
read and no re-ingest.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import re
import secrets
import sys
import time
import urllib.parse
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "data" / "episodes.json"
OUT = ROOT / "data" / "broadcast_links.json"
API = "https://api.x.com/2"

ENV = dotenv_values(ROOT / ".env")
_BROADCAST = re.compile(r"https://x\.com/i/broadcasts/([A-Za-z0-9]+)")


def header(method: str, url: str, params: dict) -> str:
    oauth = {
        "oauth_consumer_key": ENV["X_API_KEY"],
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": ENV["X_ACCESS_TOKEN"],
        "oauth_version": "1.0",
    }
    q = urllib.parse.quote
    joined = "&".join(f"{q(k, safe='')}={q(str(v), safe='')}"
                      for k, v in sorted({**oauth, **params}.items()))
    base = "&".join([method.upper(), q(url, safe=""), q(joined, safe="")])
    key = f"{q(ENV['X_API_SECRET'], safe='')}&{q(ENV['X_ACCESS_SECRET'], safe='')}"
    oauth["oauth_signature"] = base64.b64encode(
        hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()).decode()
    return "OAuth " + ", ".join(f'{q(k, safe="")}="{q(v, safe="")}"'
                                for k, v in sorted(oauth.items()))


async def broadcast_for(client: httpx.AsyncClient, status_id: str) -> str | None:
    """The /i/broadcasts/ url the show's own post links to, if any."""
    url = f"{API}/tweets/{status_id}"
    params = {"tweet.fields": "entities"}
    r = await client.get(url, params=params,
                         headers={"Authorization": header("GET", url, params)})
    if r.status_code != 200:
        print(f"    {status_id}: HTTP {r.status_code}")
        return None
    data = (r.json() or {}).get("data") or {}
    for link in (data.get("entities") or {}).get("urls") or []:
        found = _BROADCAST.search(link.get("expanded_url") or "")
        if found:
            return found.group(0)
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help=f"write {OUT.relative_to(ROOT)}")
    args = ap.parse_args()

    episodes = json.loads(EPISODES.read_text())
    broadcasts = [e for e in episodes if e["episode_id"].startswith("x-")]
    print(f"  {len(broadcasts)} X broadcasts in the archive\n")

    mapping: dict[str, str] = {}
    if OUT.exists():
        mapping = json.loads(OUT.read_text())

    async with httpx.AsyncClient(timeout=30) as client:
        for ep in broadcasts:
            eid = ep["episode_id"]
            if mapping.get(eid):
                print(f"   have  {eid}  {mapping[eid]}")
                continue
            status_id = eid.removeprefix("x-")
            link = await broadcast_for(client, status_id)
            if link:
                mapping[eid] = link
                print(f"   found {eid}  {link}")
            else:
                # Not fatal. A broadcast with no player link keeps citing
                # its status url, which is what happens today.
                print(f"   none  {eid}  ({ep.get('title', '')[:40]})")
            await asyncio.sleep(1.0)      # the read endpoint is rate limited

    print(f"\n  {len(mapping)}/{len(broadcasts)} broadcasts have a player url")
    if not args.write:
        print("  dry run. re-run with --write to save.")
        return 0
    OUT.write_text(json.dumps(mapping, indent=1, sort_keys=True))
    print(f"  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
