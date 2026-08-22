"""Announce a newly indexed episode on X.

The sync already knows the moment a new episode becomes searchable. This
turns that moment into a post, so the account stays current without anyone
remembering to do it on a Friday afternoon.

Deliberately small, and deliberately hard to misuse:

  no credentials, no posting   Absent env vars are a no-op, not an error, so
                               a local run of the sync never posts and needs
                               no configuration. Same shape as YTDLP_PROXY.
  dry run unless told twice    Posting spends real money. It happens only
                               when every credential is present AND
                               X_ANNOUNCE_LIVE is set, so a misconfigured
                               environment fails safe by printing.
  a hard cap per run           MAX_POSTS_PER_RUN bounds the blast radius of
                               a bug or a backfill. Five episodes appearing
                               at once must not become five posts and a
                               surprise bill.
  never raises                 Indexing is the job; announcing is a nicety.
                               A failure here returns False and is logged,
                               and the sync carries on.

Idempotency comes free from the pipeline rather than from bookkeeping here.
Pinecone is the source of truth for what is indexed, so an episode is "new"
exactly once, and this is only ever called after a successful ingest.

OAuth 1.0a is signed here with the standard library rather than a
dependency. The signing is forty lines of well-specified algorithm, and the
alternative was adding a package to requirements.txt that would then ship
inside the runtime container for a feature only an offline script uses.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
import urllib.parse

logger = logging.getLogger(__name__)

POST_URL = "https://api.x.com/2/tweets"

# X counts every link as 23 characters however long it really is, so the
# budget is computed against that rather than len(url).
TCO_LEN = 23
MAX_POST_CHARS = 280

# A cap, not a rate limit. One post per episode per week is the real load;
# anything above this means something has gone wrong, and it should cost a
# refusal rather than a bill.
MAX_POSTS_PER_RUN = 2

SITE = "https://search.lexthedev.com"

# Titles come from YouTube and are controlled by whoever uploads. Control
# characters and newlines are stripped before the title goes anywhere near
# a request body.
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_WS = re.compile(r"\s+")

_CREDENTIAL_VARS = (
    "X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET",
)


# ─── credentials ──────────────────────────────────────────────────────────

def credentials() -> dict[str, str] | None:
    """The four OAuth values, or None when any is missing.

    Returns None rather than raising: a machine without these configured is
    the normal case, not a broken one.
    """
    found = {name: os.environ.get(name, "").strip() for name in _CREDENTIAL_VARS}
    missing = [name for name, value in found.items() if not value]
    if missing:
        # Names only. The values are secrets and never reach a log line.
        logger.debug("X announce disabled; missing %s", ", ".join(missing))
        return None
    return found


def enabled() -> bool:
    """Whether a real post should be attempted.

    Two independent conditions, because one of them can be satisfied by
    accident. Credentials can be inherited from an environment; the explicit
    flag cannot.
    """
    return credentials() is not None and os.environ.get("X_ANNOUNCE_LIVE") == "1"


# ─── OAuth 1.0a ───────────────────────────────────────────────────────────

def _quote(value: str) -> str:
    """RFC 3986 percent-encoding.

    OAuth's spec is stricter than urlencode's default: every character
    outside the unreserved set is encoded, including the ones quote()
    normally leaves alone. Getting this wrong produces a valid-looking
    signature that the server rejects.
    """
    return urllib.parse.quote(str(value), safe="~-._")


def signature_base(method: str, url: str, params: dict[str, str]) -> str:
    """The string that gets signed: METHOD&url&sorted-encoded-params.

    Only OAuth parameters are signed for a JSON request. A JSON body is not
    form data, so it takes no part in the signature.
    """
    encoded = sorted(
        (_quote(k), _quote(v)) for k, v in params.items()
    )
    joined = "&".join(f"{k}={v}" for k, v in encoded)
    return f"{method.upper()}&{_quote(url)}&{_quote(joined)}"


def sign(base: str, consumer_secret: str, token_secret: str) -> str:
    key = f"{_quote(consumer_secret)}&{_quote(token_secret)}".encode()
    digest = hmac.new(key, base.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def auth_header(method: str, url: str, creds: dict[str, str],
                *, nonce: str | None = None,
                timestamp: str | None = None) -> str:
    """A complete OAuth 1.0a Authorization header.

    nonce and timestamp are injectable so the signing can be tested against
    fixed values; in real use they are generated fresh per request, which is
    what stops a captured request being replayed.
    """
    params = {
        "oauth_consumer_key": creds["X_API_KEY"],
        "oauth_nonce": nonce or secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": timestamp or str(int(time.time())),
        "oauth_token": creds["X_ACCESS_TOKEN"],
        "oauth_version": "1.0",
    }
    params["oauth_signature"] = sign(
        signature_base(method, url, params),
        creds["X_API_SECRET"], creds["X_ACCESS_SECRET"],
    )
    inner = ", ".join(
        f'{_quote(k)}="{_quote(v)}"' for k, v in sorted(params.items())
    )
    return f"OAuth {inner}"


# ─── the post ─────────────────────────────────────────────────────────────

def clean_title(title: str) -> str:
    return _WS.sub(" ", _CONTROL.sub(" ", str(title or ""))).strip()


def compose(title: str, url: str = SITE) -> str:
    """The announcement, built to fit inside X's limit.

    The title is the only variable-length part, so it is the part that gets
    trimmed. Budget is computed with the link counted as 23 characters,
    which is what X charges for a link of any length.
    """
    head = "new episode indexed\n\n"
    tail = f"\n\nask it anything and jump to the exact second\n{url}"
    room = MAX_POST_CHARS - len(head) - (len(tail) - len(url) + TCO_LEN)

    title = clean_title(title)
    if len(title) > room:
        title = title[:max(room - 1, 0)].rstrip() + "…"
    return f"{head}{title}{tail}"


def post(text: str, creds: dict[str, str], *, timeout: float = 20.0) -> bool:
    """Publish one post. Returns success; never raises."""
    import httpx

    try:
        response = httpx.post(
            POST_URL,
            json={"text": text},
            headers={
                "Authorization": auth_header("POST", POST_URL, creds),
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("X post failed to send: %s", type(exc).__name__)
        return False

    if response.status_code in (200, 201):
        return True
    # The body can echo request details, so log the status and a short,
    # bounded excerpt rather than the whole response.
    logger.warning("X post rejected (%s): %s",
                   response.status_code, response.text[:200])
    return False


def announce(title: str, *, sent: int = 0) -> bool:
    """Announce one episode. The single entry point the sync should call.

    `sent` is how many posts this run has already made, so the cap holds
    across a run that ingests several episodes.
    """
    if sent >= MAX_POSTS_PER_RUN:
        logger.warning("X announce skipped: already posted %d this run", sent)
        return False

    text = compose(title)
    creds = credentials()

    if not enabled():
        why = "no credentials" if creds is None else "X_ANNOUNCE_LIVE not set"
        logger.info("X announce dry run (%s):\n%s", why, text)
        return False

    ok = post(text, creds)  # type: ignore[arg-type]
    logger.info("X announce %s", "posted" if ok else "failed")
    return ok
