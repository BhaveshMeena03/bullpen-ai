"""Do the example questions on the pages actually get answered?

    python scripts/check_chips.py [--base https://search.lexthedev.com]

The chips are the first thing anyone clicks, and a chip that returns "I
couldn't find that" is the worst possible first impression: the visitor
concludes the tool does not work, and they are not wrong to.

This already happened once. "what does ansem look for before he buys"
shipped as a default chip, worked when it was chosen, and later stopped —
the corpus has plenty about why he launched a coin and nothing clean about
what he screens for before buying one, so the honest answer became a
refusal. Nothing failed loudly; the chip just quietly became a dead end,
and it was found by a stranger testing the site rather than by us.

Run this after changing chips, after a re-index, and before showing the
site to anyone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each page, the endpoint its chips hit, and the JSON field holding the
# question.
PAGES = [
    ("demo/podcast.html", "https://search.lexthedev.com",
     "/v1/podcast/search", "query"),
    ("demo/concierge.html", "https://concierge.lexthedev.com",
     "/v1/chat", "message"),
    ("demo/clawpump.html", "https://concierge.lexthedev.com",
     "/v1/clawpump/chat", "message"),
]

# A chip is dead if the answer opens with one of these. Matched at the start
# only: an answer that mentions partway through that one detail is missing is
# still a useful answer.
_DEAD = (
    "i couldn't find",
    "i could not find",
    "i don't have anything",
    "i'm ready to answer",
    "i don't have that",
)

_CHIP = re.compile(r'<button class="chip">([^<]+)</button>')
_STARTER = re.compile(r'var STARTERS = \[(.*?)\];', re.S)


def questions_on(page: str) -> list[str]:
    html = (ROOT / page).read_text()
    found = _CHIP.findall(html)
    m = _STARTER.search(html)
    if m:
        found += re.findall(r'"([^"]+)"', m.group(1))
    return found


def ask(base: str, path: str, field: str, q: str) -> str:
    req = urllib.request.Request(
        base + path, data=json.dumps({field: q}).encode(),
        headers={"content-type": "application/json"})
    # Retries generously because this is usually run alongside the other
    # checks, and together they ask far more questions per minute than the
    # rate limiter is tuned for. A 429 is this harness being impatient, not
    # the service being broken — reporting it as a dead chip would send
    # someone hunting a bug that does not exist. That already happened once.
    for attempt in range(8):
        try:
            body = urllib.request.urlopen(req, timeout=180).read()
            # strict=False: transcript text can carry raw control characters.
            return json.loads(body, strict=False).get("answer", "")
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                return f"<HTTP {exc.code}>"
            time.sleep(10 * (attempt + 1))
    return "<rate limited>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="override every page's base URL")
    args = ap.parse_args()

    dead = 0
    unreachable = 0
    for page, default_base, path, field in PAGES:
        base = (args.base or default_base).rstrip("/")
        qs = questions_on(page)
        print(f"\n{page}  ({len(qs)} example questions)")
        for q in qs:
            answer = ask(base, path, field, q)
            low = answer.strip().lower()
            if answer.startswith("<"):
                # Never asked. Distinct from "asked and got a shrug", and
                # conflating them turns a busy limiter into a false alarm.
                unreachable += 1
                state = "SKIP"
            elif any(low.startswith(m) for m in _DEAD):
                dead += 1
                state = "DEAD"
            else:
                state = "ok  "
            print(f"  {state}  {q[:52]:54s} "
                  f"{answer[:70].replace(chr(10), ' ')}")

    if unreachable:
        print(f"\n{unreachable} question(s) could not be asked at all "
              f"(rate limited or HTTP error) — not a verdict on the chip.")
    if dead:
        print(f"\n{dead} example question(s) do not get answered. "
              f"Replace them or fix retrieval before shipping.")
        sys.exit(1)
    if unreachable:
        sys.exit(2)      # distinct code: inconclusive, not failing
    print("\nevery example question gets a real answer")


if __name__ == "__main__":
    main()
