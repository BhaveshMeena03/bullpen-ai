"""Run the X mention bot.

    .venv/bin/python scripts/run_x_bot.py --once --dry-run   # read, never post
    .venv/bin/python scripts/run_x_bot.py --once             # one real cycle
    .venv/bin/python scripts/run_x_bot.py                    # the loop

Start with --dry-run. It does the full round trip — reads mentions, asks the
index, builds the reply — and stops before posting, which is the only step
that is public and irreversible. The read is what proves the credentials and
signing work; the post is what you cannot take back.

Credentials come from the environment (see .env.example). Never paste them
into a shell that records history:

    read -rs X_ACCESS_SECRET && export X_ACCESS_SECRET

Costs, from https://docs.x.com/x-api/getting-started/pricing:

    reading a mention   $0.001      answering it   ~$0.008 (Anthropic)
    posting a reply     $0.015      with a URL     $0.200

So roughly $0.024 per answered question, or about $36/month at fifty a day.
Every run prints what it spent.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.podcast import PodcastIndex  # noqa: E402
from app.x_api import OutOfCreditsError, XClient, XCredentials  # noqa: E402
from app.x_bot import MentionBot  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger("x_bot")


def build(dry_run: bool, cap: int | None, links: bool | None):
    settings = get_settings()
    missing = [
        name for name, value in (
            ("X_API_KEY", settings.x_api_key),
            ("X_API_SECRET", settings.x_api_secret),
            ("X_ACCESS_TOKEN", settings.x_access_token),
            ("X_ACCESS_SECRET", settings.x_access_secret),
            ("X_BOT_USER_ID", settings.x_bot_user_id),
        ) if not value
    ]
    if missing:
        sys.exit("missing credentials: " + ", ".join(missing)
                 + "\nsee .env.example")

    client = XClient(
        XCredentials(settings.x_api_key, settings.x_api_secret,
                     settings.x_access_token, settings.x_access_secret),
        bot_user_id=settings.x_bot_user_id,
        dry_run=dry_run,
    )
    bot = MentionBot(
        client, PodcastIndex(),
        daily_reply_cap=cap if cap is not None else settings.x_bot_daily_reply_cap,
        include_links=(links if links is not None
                       else settings.x_bot_include_links),
        contract_address=settings.x_bot_contract_address,
    )
    return settings, client, bot


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="one poll cycle, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="read and compose, but never post")
    ap.add_argument("--cap", type=int,
                    help="override the daily reply cap")
    ap.add_argument("--links", action="store_true",
                    help="include deep links — costs $0.200 per reply "
                         "instead of $0.015")
    args = ap.parse_args()

    settings, client, bot = build(args.dry_run, args.cap,
                                  True if args.links else None)

    if not args.dry_run and not args.once and not settings.x_bot_enabled:
        # The loop posts unattended. Requiring an explicit switch means
        # credentials sitting in the environment are never enough on their
        # own to start replying in public.
        sys.exit("X_BOT_ENABLED is not set — refusing to run the live loop. "
                 "Use --once or --dry-run while testing.")

    mode = "DRY RUN — nothing will be posted" if args.dry_run else "LIVE"
    log.info("%s · cap %d/day · links %s", mode, bot.cap,
             "ON ($0.200/reply)" if bot.include_links else "off ($0.015)")

    try:
        while True:
            today = time.strftime("%Y-%m-%d", time.gmtime())
            posted = await bot.tick(today)
            if posted:
                log.info("posted %d repl%s · $%.3f spent this process",
                         posted, "y" if posted == 1 else "ies",
                         client.spent_usd)
            if args.once:
                break
            await asyncio.sleep(
                MentionBot.pause_seconds(settings.x_bot_poll_seconds))
    except OutOfCreditsError as exc:
        # Certain to happen eventually and not a bug, so it exits with a
        # sentence rather than a traceback. Retrying would only spend the
        # next poll on the same refusal.
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.info("stopping")

    log.info("X spend this process: $%.3f · replies today: %d",
             client.spent_usd, bot.state.replies_today)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
