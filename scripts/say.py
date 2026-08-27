"""Post from the bot account on purpose, rather than in answer to a mention.

    .venv/bin/python scripts/say.py about
    .venv/bin/python scripts/say.py about --yes
    .venv/bin/python scripts/say.py automated --reply-to 2092822509368607177
    .venv/bin/python scripts/say.py --text "your own words here" --yes

Everything else the account posts is a reply the bot composed itself. This
is for the times you want the description in a thread without waiting for
somebody to ask — the "what is this" answer, the automation disclosure,
the contract address.

Nothing is sent without --yes. Without it the text is printed exactly as
it would appear, with its length and what it will cost, which is the
whole point: a public post is not something to discover after the fact.

Two things worth knowing before you use it:

A standalone post is not a reply, so X's February 2026 restriction does
not apply — but that restriction is also the reason replies are cheap. A
post carrying a URL may be billed at the published $0.200 rate rather
than the $0.015 the mention-gated replies have been charged, because the
measurement that showed $0.015 was taken on replies. The estimate below
assumes the worst so a surprise is on the good side.

--reply-to only works where the account you are answering has mentioned
or quoted the bot. Anywhere else X returns 403, and the script says so
rather than failing silently.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.x_api import (  # noqa: E402
    _URL_SHAPED,
    PRICE_POST,
    PRICE_POST_WITH_URL,
    XClient,
    XCredentials,
)
from app.x_bot import (  # noqa: E402
    about_answer,
    automation_answer,
    pinned_answer,
    weighted_length,
)


def canned(name: str, settings) -> str | None:
    """The fixed messages, built from the same code the bot replies with.

    Written once, in app/x_bot.py, so a description posted by hand and one
    the bot gives when asked cannot drift apart.
    """
    site = settings.x_bot_site
    if name == "about":
        return about_answer("what is this", site)
    if name == "automated":
        return automation_answer("are you a bot", site)
    if name == "ca":
        return pinned_answer("ca?", settings.x_bot_contract_address,
                             settings.x_bot_token_label)
    return None


CHOICES = ("about", "automated", "ca")


def _post_id(value: str) -> str:
    """Accept a bare id or a full x.com URL."""
    value = value.strip().rstrip("/")
    return value.rsplit("/", 1)[-1].split("?")[0]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?", choices=CHOICES,
                    help="which fixed message to post")
    ap.add_argument("--text", help="post your own words instead")
    ap.add_argument("--reply-to", metavar="ID_OR_URL",
                    help="reply to a post that mentioned the bot")
    ap.add_argument("--yes", action="store_true",
                    help="actually send it — without this, nothing is posted")
    args = ap.parse_args()

    if bool(args.message) == bool(args.text):
        ap.error("give exactly one of a message name or --text")

    settings = get_settings()
    text = args.text or canned(args.message, settings)
    if not text:
        print(f"nothing to say for {args.message!r}")
        return 2

    has_link = bool(_URL_SHAPED.search(text))
    length = weighted_length(text)
    cost = PRICE_POST_WITH_URL if has_link else PRICE_POST

    print("\n" + "=" * 66)
    for line in text.splitlines():
        print(f"  {line}")
    print("=" * 66)
    print(f"  {length} characters as X counts them")
    print(f"  {'a reply to ' + _post_id(args.reply_to) if args.reply_to else 'a standalone post'}")
    print(f"  up to ${cost:.3f}"
          + ("  (carries a link — may be billed at the URL rate)"
             if has_link else ""))

    if not args.yes:
        print("\n  Nothing was posted. Add --yes to send it.\n")
        return 0

    client = XClient(
        XCredentials(settings.x_api_key, settings.x_api_secret,
                     settings.x_access_token, settings.x_access_secret),
        bot_user_id=settings.x_bot_user_id,
    )
    if args.reply_to:
        posted = await client.reply(text, _post_id(args.reply_to),
                                    allow_link=has_link)
    else:
        posted = await client.post(text, allow_link=has_link)

    if not posted:
        print("\n  X refused it. For a reply, the usual cause is that the "
              "account being answered has not mentioned the bot.\n")
        return 1
    print(f"\n  posted: https://x.com/mbubbleSearch/status/{posted}")
    print(f"  spent this run: ${client.spent_usd:.3f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
