"""Rate limiting shared by every bot in this package.

Both bots call a backend that spends model tokens per request, so both need
the same two guards: a per-user cooldown and a bot-wide ceiling. Keeping them
here means a second bot inherits the budget protection instead of
reimplementing it slightly differently.
"""

from __future__ import annotations


class Cooldown:
    """Per-user cooldown. Discord has its own limits, but this specifically
    guards the backend's model spend from a single user spamming a command."""

    def __init__(self, seconds: float):
        self._seconds = seconds
        self._last: dict[int, float] = {}

    def retry_after(self, user_id: int, now: float) -> float:
        last = self._last.get(user_id, 0.0)
        remaining = self._seconds - (now - last)
        return max(0.0, remaining)

    def stamp(self, user_id: int, now: float) -> None:
        self._last[user_id] = now
        if len(self._last) > 10_000:  # bounded
            oldest = min(self._last, key=self._last.get)
            del self._last[oldest]


class GlobalThrottle:
    """A hard ceiling on requests/minute across the ENTIRE bot — the budget
    backstop. Per-user cooldowns don't stop a coordinated spam raid in a big
    server; this caps total model spend regardless of how many users pile on.
    Token bucket: `per_min` tokens, refilled continuously."""

    def __init__(self, per_min: float):
        self._rate = per_min / 60.0
        self._burst = max(1.0, per_min)
        self._tokens = self._burst
        self._last = 0.0

    def allow(self, now: float) -> bool:
        if self._last == 0.0:
            self._last = now
        self._tokens = min(self._burst, self._tokens + (now - self._last) * self._rate)
        self._last = now
        if self._tokens < 1.0:
            return False
        self._tokens -= 1.0
        return True
