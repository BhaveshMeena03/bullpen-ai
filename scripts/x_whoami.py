"""Check X credentials and print the numeric user id the bot needs.

    .venv/bin/python scripts/x_whoami.py

The first thing to run after creating the app at console.x.com. It calls
GET /2/users/me, which proves in one shot that the four keys are right, that
the OAuth signature is being built correctly, and that the tokens belong to
the account you meant — then prints the numeric id for X_BOT_USER_ID, which
the mentions endpoint requires instead of the @handle.

Costs $0.001. Nothing is posted.

Reading the failures:

  401   the signature or the keys. If the keys were re-copied and it still
        fails, the usual cause is a stray character from the dashboard —
        config.py strips those, so check for a swapped key/secret pair.
  403   authentication worked but this app is not allowed to do it. For a
        write, that means the tokens carry Read-only permission: set the app
        to Read and Write, then REGENERATE the access token and secret.
        Tokens keep the permission they were minted with, so changing the
        setting alone does not upgrade them.
  429   rate limited; wait and retry.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.x_api import XClient, XCredentials  # noqa: E402


async def main() -> int:
    settings = get_settings()
    keys = {
        "X_API_KEY": settings.x_api_key,
        "X_API_SECRET": settings.x_api_secret,
        "X_ACCESS_TOKEN": settings.x_access_token,
        "X_ACCESS_SECRET": settings.x_access_secret,
    }
    missing = [name for name, value in keys.items() if not value]
    if missing:
        print("missing: " + ", ".join(missing))
        print("\nSet them without putting secrets in your shell history:")
        for name in missing:
            print(f"    read -rs {name} && export {name}")
        return 1

    for name, value in keys.items():
        print(f"  {name:<16} {value[:6]}…{value[-4:]}  ({len(value)} chars)")

    client = XClient(
        XCredentials(settings.x_api_key, settings.x_api_secret,
                     settings.x_access_token, settings.x_access_secret),
        # Not known yet — that is the point of this script.
        bot_user_id="",
    )
    try:
        me = await client.whoami()
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        print(f"\n  FAILED — HTTP {code}")
        print(f"  {exc.response.text[:300]}")
        if code == 401:
            print("\n  401 is the signature or the keys themselves. Check "
                  "that the API key/secret pair and the access token/secret "
                  "pair were not swapped.")
        elif code == 403:
            print("\n  403 means the app is not permitted. Set the app to "
                  "Read and Write, then REGENERATE the access token and "
                  "secret — tokens keep whatever permission they were "
                  "created with.")
        return 1

    print(f"\n  authenticated as @{me.get('username')} ({me.get('name')})")
    print(f"  spent ${client.spent_usd:.3f}\n")
    print("  Add this to .env:\n")
    print(f"    X_BOT_USER_ID={me.get('id')}\n")
    handle = me.get("username") or ""
    if len(handle) > 15:
        # Cannot actually happen — X enforces it at signup — but if it ever
        # does, the mentions endpoint is not the thing to debug.
        print(f"  note: @{handle} is {len(handle)} characters, over X's 15")
    print("  Then read, without posting anything:")
    print("    .venv/bin/python scripts/run_x_bot.py --once --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
