"""Minimal X API v2 client for the mention bot.

Two operations only: read the account's own mentions, and reply to one. No
SDK, because the surface needed here is small and every X dependency is a
thing that can break a bot that posts publicly under someone's name.

Every call costs real money, so the pricing is written down next to the code
that spends it (https://docs.x.com/x-api/getting-started/pricing):

    owned read   GET /2/users/{id}/mentions      $0.001 per resource
    post create  POST /2/tweets                  $0.015 per request
    post create  POST /2/tweets, with a URL      $0.200 per request

That last row is the whole design. A reply containing a link costs thirteen
times one that does not, which turns a $36/month bot into a $313/month one
at fifty mentions a day. So replies carry the quote and the timestamp, and
the link lives in the profile bio instead — cheaper, and a better reply.

X does not document what counts as "a URL" — nothing about bare domains,
t.co, or media. Rather than guess at their detector, `assert_linkless`
refuses anything URL-shaped, which keeps the cheap rate whichever way their
rule actually works.

Reads are deduplicated within a UTC day: fetching the same mention again
costs nothing until midnight, so polling often is fine.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

API = "https://api.x.com/2"

# Published per-call prices, so a run can report what it actually spent
# rather than an estimate someone has to look up.
PRICE_OWNED_READ = 0.001
# An expanded author object. Undocumented whether X bills expansions
# separately, so it is counted — over-counting only makes the daily spend
# ceiling arrive early, and the ceiling arriving early is the safe error.
PRICE_USER_READ = 0.010
PRICE_POST = 0.015
PRICE_POST_WITH_URL = 0.200

# Anything a link detector might plausibly catch. Deliberately broader than
# "starts with http": a bare domain in a quoted transcript line would still
# be a URL to X, and being wrong here costs 13x per reply.
_URL_SHAPED = re.compile(
    r"""(?xi)
    (?: https?://                       # explicit scheme
      | www\.                           # conventional prefix
      | \b[a-z0-9][a-z0-9-]*\.          # bare domain: name.tld
        (?: com|org|net|io|ai|co|xyz|app|dev|tech|gg|so|fun|me|tv
          | link|sh|to|us|uk|info|biz|eth|sol|fi|xx )\b
    )""")


class LinkInReplyError(RuntimeError):
    """Raised rather than paying 13x for a reply nobody meant to make cost that."""


class OutOfCreditsError(RuntimeError):
    """X returned 402: the account's credit balance is spent.

    Its own category because it is the one failure that is certain to happen
    eventually and is not a bug. Every billed call returns it, so without
    this the bot dies on an unhandled HTTPStatusError — a stack trace at 3am
    that reads like a crash when the answer is "top up the balance".

    Worth knowing: GET /2/users/me is not billed, so credentials can verify
    perfectly against a zero balance and the first real call still fails.
    That is confusing enough to deserve saying out loud.
    """


def strip_urls(text: str) -> str:
    """Drop URL-shaped tokens from text that came from a transcript.

    Guests read links out loud and Whisper writes them down, so a quote can
    carry one without anyone intending it.

    Line by line, because `text.split()` splits on every kind of whitespace
    and rejoining with spaces silently flattens paragraphs. That did not
    matter while replies were two sentences; it turned a three-thousand
    character episode summary into one unreadable block.
    """
    return "\n".join(
        " ".join(w for w in line.split() if not _URL_SHAPED.search(w))
        for line in (text or "").splitlines()
    )


def assert_linkless(text: str) -> None:
    found = _URL_SHAPED.search(text)
    if found:
        raise LinkInReplyError(
            f"reply contains {found.group(0)!r}, which would cost "
            f"${PRICE_POST_WITH_URL:.3f} instead of ${PRICE_POST:.3f}"
        )


def _raise_if_out_of_credits(response: httpx.Response) -> None:
    """Turn X's 402 into something a log reader can act on."""
    if response.status_code == 402:
        raise OutOfCreditsError(
            "X returned 402 Payment Required — the credit balance is spent. "
            "Top up at console.x.com (Billing -> Credits). Note that "
            "GET /2/users/me is not billed, so scripts/x_whoami.py can "
            "succeed while every real call fails."
        )


@dataclass(frozen=True)
class Mention:
    """One post that tagged the bot."""

    id: str
    text: str
    author_id: str
    conversation_id: str
    # X's paid-tier badge. Worth knowing that it means "pays for Premium"
    # rather than "is who they claim to be" — it filters throwaway accounts,
    # not bad intentions.
    author_verified: bool = False
    author_verified_type: str = "none"
    # ISO-8601 from X. Used by the cold start: a deploy wipes the state file
    # on an ephemeral disk, and "skip everything pending" would then drop a
    # question asked a minute earlier.
    created_at: str = ""


class XCredentials:
    """OAuth 1.0a user-context keys.

    User context rather than an app-only bearer token because the bot posts
    as itself, and app-only cannot write.
    """

    def __init__(self, consumer_key: str, consumer_secret: str,
                 access_token: str, access_secret: str) -> None:
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret

    def header(self, method: str, url: str,
               params: dict[str, str] | None = None) -> str:
        """An OAuth 1.0a Authorization header, signed HMAC-SHA1.

        Implemented here rather than pulling in requests-oauthlib: it is
        forty lines of well-specified signing, and the project keeps its
        dependency list short on purpose.

        Only query parameters are signed. JSON bodies are not part of the
        signature base string under OAuth 1.0a, which is what X expects for
        the v2 endpoints used here.
        """
        oauth = {
            "oauth_consumer_key": self.consumer_key,
            "oauth_nonce": secrets.token_hex(16),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_token": self.access_token,
            "oauth_version": "1.0",
        }
        quote = urllib.parse.quote
        signing = {**oauth, **(params or {})}
        joined = "&".join(
            f"{quote(k, safe='')}={quote(str(v), safe='')}"
            for k, v in sorted(signing.items())
        )
        base = "&".join(
            [method.upper(), quote(url, safe=""), quote(joined, safe="")])
        key = f"{quote(self.consumer_secret, safe='')}&" \
              f"{quote(self.access_secret, safe='')}"
        signature = base64.b64encode(
            hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()
        ).decode()
        oauth["oauth_signature"] = signature
        return "OAuth " + ", ".join(
            f'{quote(k, safe="")}="{quote(v, safe="")}"'
            for k, v in sorted(oauth.items())
        )


class XClient:
    """The two calls the bot makes, and what they cost."""

    def __init__(self, credentials: XCredentials, bot_user_id: str,
                 dry_run: bool = False) -> None:
        self._credentials = credentials
        self.bot_user_id = bot_user_id
        # dry_run reads normally but never posts. The read is what tells you
        # the wiring works; the post is the part that is public and paid.
        self._dry_run = dry_run
        self.spent_usd = 0.0

    async def whoami(self) -> dict:
        """The authenticated account: {"id", "username", "name"}.

        The first call worth making with new credentials. It proves four
        things at once — the keys are right, the signature base string is
        being built correctly, the tokens belong to the account you think,
        and it hands back the numeric id that X_BOT_USER_ID wants (the
        mentions endpoint takes an id, not a handle).

        An owned read, so $0.001.
        """
        url = f"{API}/users/me"
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.get(
                url,
                headers={"Authorization":
                         self._credentials.header("GET", url)},
            )
        _raise_if_out_of_credits(response)
        response.raise_for_status()
        self.spent_usd += PRICE_OWNED_READ
        return response.json().get("data") or {}

    async def mentions(self, since_id: str | None = None,
                       limit: int = 20) -> list[Mention]:
        """Posts that tagged the bot, newest last.

        `since_id` is what keeps this from re-reading — and the bot from
        re-answering — the same backlog after a restart.
        """
        url = f"{API}/users/{self.bot_user_id}/mentions"
        params = {
            "max_results": str(max(5, min(limit, 100))),
            "tweet.fields": "author_id,conversation_id,created_at",
            # The author comes back in the same response rather than needing
            # a lookup per mention. Repeat askers are deduplicated within the
            # UTC day like everything else, so a regular costs nothing after
            # the first time.
            "expansions": "author_id",
            "user.fields": "verified,verified_type",
        }
        if since_id:
            params["since_id"] = since_id
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.get(
                url, params=params,
                headers={"Authorization":
                         self._credentials.header("GET", url, params)},
            )
        if response.status_code == 429:
            logger.warning("X rate limited the mentions read; backing off")
            return []
        _raise_if_out_of_credits(response)
        response.raise_for_status()
        body = response.json()
        found = body.get("data") or []
        # Charged per resource returned, and deduplicated for the rest of
        # the UTC day, so an empty poll is free.
        self.spent_usd += len(found) * PRICE_OWNED_READ
        authors = {u["id"]: u
                   for u in (body.get("includes", {}).get("users") or [])}
        self.spent_usd += len(authors) * PRICE_USER_READ
        return [
            Mention(
                id=m["id"], text=m.get("text", ""),
                author_id=m.get("author_id", ""),
                conversation_id=m.get("conversation_id", m["id"]),
                created_at=m.get("created_at", ""),
                author_verified=bool(
                    authors.get(m.get("author_id", ""), {}).get("verified")),
                author_verified_type=(
                    authors.get(m.get("author_id", ""), {})
                    .get("verified_type") or "none"),
            )
            for m in reversed(found)
        ]

    async def reply(self, text: str, to_post_id: str,
                    allow_link: bool = False) -> str | None:
        """Reply to the post that tagged the bot. Returns the new post id.

        Permitted because the author mentioned the bot first: X restricted
        programmatic replies in February 2026 to exactly that case, which is
        why tag-to-ask still works when generic reply bots do not.
        """
        if not allow_link:
            # The guard is against a URL nobody meant to send — one read
            # aloud in a transcript, or a model writing one unprompted. When
            # links are deliberately enabled the charge is the point, so
            # asserting here refused the very mode that was switched on.
            assert_linkless(text)
        if self._dry_run:
            logger.info("[dry run] would reply to %s: %s", to_post_id, text)
            return None
        url = f"{API}/tweets"
        payload = {"text": text,
                   "reply": {"in_reply_to_tweet_id": to_post_id}}
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.post(
                url, json=payload,
                headers={"Authorization":
                         self._credentials.header("POST", url),
                         "content-type": "application/json"},
            )
        if response.status_code == 403:
            # The usual cause is the reply restriction: the author has to
            # have mentioned the bot. Worth its own message, because it
            # looks like an auth failure and is not one.
            logger.warning("X refused the reply to %s (403): %s",
                           to_post_id, response.text[:200])
            return None
        _raise_if_out_of_credits(response)
        response.raise_for_status()
        self.spent_usd += PRICE_POST_WITH_URL if allow_link else PRICE_POST
        return (response.json().get("data") or {}).get("id")
