"""Request-level protections: per-IP rate limiting and admin auth.

Both are FastAPI dependencies so they're visible in the route signature
and the OpenAPI docs, and testable in isolation.
"""

import ipaddress
import logging
import secrets
import time

from fastapi import HTTPException, Request

from .config import get_settings

logger = logging.getLogger(__name__)


def _client_host(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _is_public_ip(value: str) -> bool:
    """True for a routable internet address.

    Used to skip the host's own infrastructure when reading X-Forwarded-For.
    Anything unparseable is treated as not-public, so a garbage entry can
    never be selected as the identity to rate-limit on.
    """
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_unspecified)


class RateLimiter:
    """In-memory token bucket, keyed by client IP.

    Suitable for a single-process deployment; swap the storage for Redis
    when running multiple replicas. Defaults come from settings so limits
    are tunable per environment without code changes.
    """

    MAX_BUCKETS = 10_000  # memory backstop against IP churn/spoofing

    def __init__(self, rpm: int | None = None, burst: int | None = None):
        self._rpm = rpm
        self._burst = burst
        self._buckets: dict[str, tuple[float, float]] = {}

    def _limits(self) -> tuple[float, int]:
        rpm = self._rpm or get_settings().rate_limit_rpm
        burst = self._burst or max(5, rpm // 3)
        return rpm / 60.0, burst

    def check(self, key: str) -> bool:
        """Consume one token for `key`; False when the bucket is empty."""
        rate, burst = self._limits()
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (float(burst), now))
        tokens = min(float(burst), tokens + (now - last) * rate)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False
        if len(self._buckets) >= self.MAX_BUCKETS and key not in self._buckets:
            # Evict the single least-recently-seen bucket instead of wiping
            # them all (a full clear would momentarily reset everyone's limit
            # under IP churn). Bounds memory without lifting active throttles.
            oldest = min(self._buckets, key=lambda k: self._buckets[k][1])
            del self._buckets[oldest]
        self._buckets[key] = (tokens - 1.0, now)
        return True

    @staticmethod
    def _client_ip(request: Request) -> str:
        """Identify the caller for bucketing, without letting them choose.

        Behind Render's proxy, request.client.host is the proxy's IP for every
        user, which would collapse everyone into one bucket. So X-Forwarded-For
        has to be consulted — but it cannot be trusted naively.

        This used to read the LEFT-most hop. Proxies append, so a caller who
        sends their own X-Forwarded-For lands first in the list and the address
        Render observed is appended after it: the left-most entry is whatever
        the caller typed. Confirmed against the live service — 40 requests each
        carrying a different forged value never hit the 30/min limit, while 40
        carrying the same forged value were throttled at exactly the limit.

        Counting from the RIGHT fixes it. The last entry was written by our own
        proxy and a client cannot append past it, so with one proxy in front
        (trusted_proxy_hops=1) hop -1 is the real client.

        Worth being precise about the blast radius: the global RPM ceiling and
        the daily budget never keyed on identity, so the model-spend cap held
        throughout. What was bypassable was the per-IP fairness limit.
        """
        xff = request.headers.get("x-forwarded-for")
        if xff:
            hops = [h.strip() for h in xff.split(",") if h.strip()]
            # Walk right-to-left and take the first address that is not
            # infrastructure. Counting a fixed number of hops from either end
            # is brittle: hard-coding the left end lets the caller pick their
            # own bucket, and hard-coding the right end assumes the host
            # appends exactly one stable address. Skipping private, loopback
            # and link-local entries lands on the real client without needing
            # to know how many proxies are in front or what they write.
            for hop in reversed(hops):
                if _is_public_ip(hop):
                    return hop
            if hops:
                return hops[-1]
        return request.client.host if request.client else "unknown"

    async def __call__(self, request: Request) -> None:
        if not self.check(self._client_ip(request)):
            raise HTTPException(
                status_code=429,
                detail="Too many requests — please slow down.",
                headers={"Retry-After": "5"},
            )


class GlobalRateLimit(RateLimiter):
    """Single bucket across ALL clients — a spend ceiling for the model API.

    Per-IP limiting stops one abuser; this stops a botnet from burning the
    Anthropic budget. Tune via GLOBAL_RATE_LIMIT_RPM.
    """

    def _limits(self) -> tuple[float, int]:
        rpm = get_settings().global_rate_limit_rpm
        return rpm / 60.0, max(10, rpm // 2)

    async def __call__(self, request: Request) -> None:
        if not self.check("global"):
            raise HTTPException(
                status_code=429,
                detail="The service is busy right now — try again in a minute.",
                headers={"Retry-After": "30"},
            )


class DailyBudget:
    """A hard ceiling on model-backed requests per UTC day.

    The rate limiters bound requests per *minute*, which stops a burst but
    says nothing about a slow drain: 25 requests a minute sits inside every
    existing limit and still reaches ~36,000 chats in a day. Since traffic
    here is unpredictable, the protection that matters is the one that holds
    without anyone watching it.

    Counting requests rather than tokens is deliberate. Per-request cost is
    already bounded — history is capped and so is max_tokens — so a request
    count maps to a bounded worst case, and it needs no bookkeeping of what
    the provider actually billed.

    In-memory, matching the rate limiters: correct for the single-process
    deployment this runs on, and it resets on restart. Move it to Redis at
    the same time as those, if there is ever a second replica.
    """

    def __init__(self, limit: int | None = None):
        self._limit = limit
        self._day: str | None = None
        self._count = 0

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _ceiling(self) -> int:
        return self._limit if self._limit is not None else get_settings().daily_request_budget

    def check(self) -> bool:
        limit = self._ceiling()
        if limit <= 0:          # 0 disables the cap
            return True
        today = self._today()
        if today != self._day:  # UTC rollover
            self._day, self._count = today, 0
        if self._count >= limit:
            return False
        self._count += 1
        return True

    def state(self) -> dict:
        return {"day": self._day, "used": self._count, "limit": self._ceiling()}

    async def __call__(self, request: Request) -> None:
        if not self.check():
            logger.error(
                "Daily request budget exhausted (%s) — refusing model calls.",
                self.state(),
            )
            raise HTTPException(
                status_code=503,
                detail=("The assistant has reached its daily limit. "
                        "It will reset at midnight UTC."),
                headers={"Retry-After": "3600"},
            )


# Shared limiters for the public chat/search endpoints (both applied).
public_rate_limit = RateLimiter()
global_rate_limit = GlobalRateLimit()
daily_budget = DailyBudget()


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def require_admin(request: Request) -> None:
    """Guard for ingestion endpoints.

    Callers must send ADMIN_TOKEN as X-Admin-Token.

    With no token configured this FAILS CLOSED for anything off-loopback.
    It used to return early and wave the request through, so an environment
    that lost the variable — a new deploy, a cleared dashboard field, a fresh
    clone — silently published /v1/ingest and /v1/podcast/ingest to the
    internet. Those endpoints rewrite the knowledge base and spend model
    budget, and nothing in the response would have told you the door was open.
    Local development still works unauthenticated because it comes from
    127.0.0.1.
    """
    expected = get_settings().admin_token
    if not expected:
        if _is_loopback(request):
            return
        logger.error(
            "Admin endpoint hit from %s with no ADMIN_TOKEN set — refusing.",
            _client_host(request),
        )
        raise HTTPException(
            status_code=503,
            detail="Admin endpoints are disabled: server has no ADMIN_TOKEN configured.",
        )
    provided = request.headers.get("x-admin-token", "")
    # Compare bytes, not str: Starlette decodes headers as latin-1, so a
    # non-ASCII header byte would make compare_digest(str, str) raise a
    # TypeError (surfacing as a 500). Encoding both sides avoids that and
    # still runs in constant time.
    if not secrets.compare_digest(
        provided.encode("utf-8", "ignore"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")
