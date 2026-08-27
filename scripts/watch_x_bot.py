"""Say when the mention bot has stopped answering.

    .venv/bin/python scripts/watch_x_bot.py
    .venv/bin/python scripts/watch_x_bot.py --once
    .venv/bin/python scripts/watch_x_bot.py --expect 189a13a

The failure this is for: a build failed, Render kept serving the previous
image, and for twenty minutes the URL answered, /healthz returned ok, and
the logs were quiet — while the bot replied to nobody. Every signal we had
said healthy. The only one that would have caught it is which commit is
actually running, which is why --expect exists: pass the commit you just
pushed and it says whether that is the code answering.

Exits non-zero when something is wrong, so it can be a cron line or a
shell one-liner rather than something anybody has to remember to read.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

DEFAULT = "https://search.lexthedev.com/x-bot/status"


def check(url: str, expect: str | None) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {url}"
    except Exception as exc:                                  # noqa: BLE001
        # Unreachable is itself the alarm — this runs from outside the box
        # precisely so that the box being gone is visible.
        return False, f"unreachable: {exc}"

    version = body.get("version", "?")
    if body.get("note"):
        return False, f"no heartbeat (build {version}) — {body['note']}"
    if not body.get("enabled"):
        return True, f"bot is switched off (build {version})"
    if expect and not version.startswith(expect[:7]):
        return False, (f"running build {version}, expected {expect[:7]} — "
                       f"the deploy did not land")
    if not body.get("healthy"):
        since = body.get("seconds_since_poll")
        return False, (f"NOT POLLING — last cycle {since}s ago "
                       f"(build {version})")
    errors = body.get("consecutive_errors") or 0
    if errors >= 3:
        return False, f"{errors} failed polls in a row (build {version})"
    if not body.get("highlights"):
        return False, (f"highlight pool is empty (build {version}) — "
                       f"compliments will get silence")

    return True, (f"ok · build {version} · polled "
                  f"{body.get('seconds_since_poll')}s ago · "
                  f"{body.get('replies')} replies · "
                  f"{body.get('highlights')} highlights")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=120.0,
                    help="seconds between checks when looping")
    ap.add_argument("--expect", help="commit that should be running")
    args = ap.parse_args()

    while True:
        ok, message = check(args.url, args.expect)
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp}  {'  ' if ok else '!! '}{message}", flush=True)
        if args.once:
            return 0 if ok else 1
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
