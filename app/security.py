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
        # Behind Render's proxy, request.client.host is the proxy's IP for
        # every user — which would collapse all clients into one bucket. Use
        # the left-most X-Forwarded-For hop (the real client) when present.
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
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


# Shared limiters for the public chat/search endpoints (both applied).
public_rate_limit = RateLimiter()
global_rate_limit = GlobalRateLimit()


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
