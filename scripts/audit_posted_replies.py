"""Check what the account actually said, not what it would say.

    .venv/bin/python scripts/audit_posted_replies.py
    .venv/bin/python scripts/audit_posted_replies.py --limit 20

verify_replies.py asks a fixed list of questions and checks the answers
it gets back. That is a test of the current build. It cannot see a reply
posted two days ago by a build that has since changed, and those are the
ones strangers are reading.

"Around 5:43 in the June 25 episode, targets were: Bitcoin $58K,
Hyperliquid $55K" was live for about two hours. Hyperliquid has never
traded within three orders of magnitude of $55,000. The number was real
and belonged to Bitcoin; the line it came from was Banks guessing at
Ansem's targets and saying so, with Ansem correcting him immediately
after. It came down only because somebody happened to ask.

Four checks per reply, run against the transcript the reply itself links
to:

  reachable   the ?t= lands inside an episode that exists
  quoted      a quoted fragment appears near that second
  speaker     a host named in the reply matches the voice label on the
              line the quote came from
  figures     every number in the reply appears near the second it cites

`figures` is the one the earlier verifier did not have, and the one that
would have caught the price targets: that reply passed episode, second,
support and speaker, and was still false.

No model calls. Everything is matched against local transcripts, so the
whole run costs the price of reading the timeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.x_api import API, XCredentials  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
SPEAKERS = ROOT / "data" / "speaker_map.json"

T_PARAM = re.compile(r"[?&]t=(\d+)s?")
YT_ID = re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})")
X_ID = re.compile(r"x\.com/\w+/status/(\d+)")
QUOTED = re.compile(r'"([^"]{12,150})"')
CITED = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b(?!\s*(?:[ap]\.?m\.?)\b)", re.I)
HOSTS = ("FaZe Banks", "Banks", "Ansem")
# "$55K", "58K", "$600", "4.5 billion"
FIGURE = re.compile(r"\$?\d[\d.,]*\s*(?:k\b|K\b|million|billion|m\b|bn\b)?", re.I)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']{4,}", text.lower())}


def load():
    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}
    speakers = json.loads(SPEAKERS.read_text()) if SPEAKERS.exists() else {}
    return episodes, speakers


def episode_for(link: str, episodes: dict) -> dict | None:
    found = YT_ID.search(link)
    if found and found.group(1) in episodes:
        return episodes[found.group(1)]
    found = X_ID.search(link)
    if found and f"x-{found.group(1)}" in episodes:
        return episodes[f"x-{found.group(1)}"]
    return None


def near(episode: dict, at: int, reach: int = 150) -> str:
    return " ".join(s.get("text", "") for s in episode["segments"]
                    if abs(s.get("t", 0) - at) <= reach).lower()


def labelled_near(episode: dict, speakers: dict, at: int,
                  reach: int = 150) -> list[tuple[str, str]]:
    """(speaker, line) for the labelled lines around `at`."""
    lab = speakers.get(episode["episode_id"], {})
    numbered = [s for s in episode["segments"] if (s.get("text") or "").strip()]
    out = []
    for i, s in enumerate(numbered):
        if abs(s.get("t", 0) - at) <= reach:
            who = lab.get(str(i))
            if who:
                out.append((who, s["text"]))
    return out


def audit(reply: str, episodes: dict, speakers: dict,
          link: str | None = None) -> list[str]:
    """`link` is the EXPANDED url from the API.

    X rewrites every link to t.co on the way out, so the posted text
    never contains the address the reply actually points at. Matching on
    the body alone resolved nothing, silently, and reported a clean run
    over ninety-two replies it had not looked at.
    """
    problems: list[str] = []
    link = link or next(iter(re.findall(r"https?://\S+", reply)), "")
    if not link:
        return problems                      # nothing to check it against

    episode = episode_for(link, episodes)
    if not episode:
        return problems                      # a link to the site, not an episode

    found = T_PARAM.search(link)
    stamps = CITED.findall(reply)
    if found:
        at = int(found.group(1))
    elif stamps:
        parts = [int(p) for p in stamps[0].split(":")]
        at = (parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3
              else parts[0] * 60 + parts[1])
    else:
        return problems

    # Two texts, because the two checks ask different questions.
    #
    # A QUOTE has to appear near the second the reply points at -- that is
    # the promise the timestamp makes.
    #
    # A FIGURE only has to be somewhere in the episode. Replies routinely
    # cover four moments and cite one: a reply about the Orangie
    # conversation cited 34:25 and mentioned "18 months early", which is
    # said at 1:23 of the same episode. Checking a number against a
    # 150-second window called a true statement a fabrication, twice.
    text = near(episode, at)
    whole = " ".join(x.get("text", "") for x in episode["segments"]).lower()
    if not text:
        problems.append(f"the link lands at {at}s, where the episode has "
                        f"no transcript")
        return problems

    for quote in QUOTED.findall(reply):
        target = words(quote)
        if len(target) < 4:
            continue
        overlap = len(target & words(text))
        if overlap < max(2, len(target) // 4):
            problems.append(f'quote not found near the cited second: '
                            f'"{quote[:56]}"')

    lines = labelled_near(episode, speakers, at)
    if lines:
        for host in ("Ansem", "Banks"):
            if not re.search(rf"\b{host}\b", reply):
                continue
            said = {w for who, line in lines
                    if (who == "Ansem") == (host == "Ansem")
                    for w in words(line)}
            for quote in QUOTED.findall(reply):
                q = words(quote)
                if len(q) < 4:
                    continue
                other = {w for who, line in lines
                         if (who == "Ansem") != (host == "Ansem")
                         for w in words(line)}
                if len(q & other) > len(q & said) and len(q & other) >= 3:
                    problems.append(
                        f'"{quote[:44]}" reads as the other host, '
                        f'not {host}')

    # Every figure in the reply should appear near the second it cites.
    # Scanned over the prose only: a URL carries a nineteen-digit post id
    # and the citation carries the timestamp, and neither is a claim.
    prose = re.sub(r"https?://\S+", " ", reply)
    prose = CITED.sub(" ", prose)
    # @CryptoExpert101 and @blknoiz06 are not figures, and neither is
    # "the August 20 episode" or "the 2013-14 cycle" -- those locate a
    # moment rather than assert a quantity.
    prose = re.sub(r"@\w+", " ", prose)
    prose = re.sub(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
                   r"[a-z]*\s+\d{1,2}\b", " ", prose)
    prose = re.sub(r"(?i)\b(?:episode|ep|#)\s*\d{1,3}\b", " ", prose)
    prose = re.sub(r"\b(?:19|20)\d{2}(?:-\d{2})?\b", " ", prose)
    for raw in FIGURE.findall(prose):
        digits = re.sub(r"[^\d.]", "", raw)
        if not digits or len(digits.rstrip(".")) < 2:
            continue
        # Whisper writes large numbers with spaces as often as commas --
        # "there's fucking 30 000 coins that launch a day" -- so both
        # separators come out before matching, or a true figure reads as
        # unsupported.
        flat = re.sub(r"(?<=\d)[ ,](?=\d)", "", whole)
        if digits.rstrip(".") in flat:
            continue
        bare = digits.split(".")[0]
        if bare and bare in flat:
            continue
        problems.append(f"figure {raw.strip()} is not near the cited second")
    return problems


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    episodes, speakers = load()
    s = get_settings()
    cred = XCredentials(s.x_api_key, s.x_api_secret,
                        s.x_access_token, s.x_access_secret)
    url = f"{API}/users/{s.x_bot_user_id}/tweets"
    params = {"max_results": str(max(5, min(args.limit, 100))),
              "tweet.fields": "created_at,note_tweet,entities,in_reply_to_user_id"}
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.get(url, params=params,
                           headers={"Authorization": cred.header("GET", url, params)})
    if r.status_code != 200:
        print(f"  could not read the timeline: HTTP {r.status_code}")
        return 2

    posts = r.json().get("data") or []
    checked = flagged = 0
    print(f"\n  {len(posts)} posts on the timeline\n")
    for p in posts:
        body = (p.get("note_tweet") or {}).get("text") or p["text"]
        expanded = [u.get("expanded_url", "") for u in
                    (p.get("entities", {}).get("urls") or [])]
        episode_link = next(
            (u for u in expanded if episode_for(u, episodes)), None)
        if not episode_link:
            continue
        checked += 1
        problems = audit(body, episodes, speakers, link=episode_link)
        if problems:
            flagged += 1
            print(f"  {'=' * 66}")
            print(f"  {p['created_at'][:16]}  "
                  f"https://x.com/mbubbleSearch/status/{p['id']}")
            for line in body.splitlines()[:3]:
                print(f"  | {line[:76]}")
            for problem in problems:
                print(f"      -> {problem}")
    print(f"\n  {checked} replies pointed at an episode we hold · "
          f"{flagged} flagged\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
