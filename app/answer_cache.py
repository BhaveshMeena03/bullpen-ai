"""Cache for whole answers, keyed on the question.

Prompt caching already cuts the cost of the *input* to a model call. This
skips the call. Support and search traffic is dominated by repeats — "what
are the fees", "how do I fund my wallet", whatever chip is on the page —
and today the thousandth person to ask pays exactly what the first did, and
waits exactly as long.

Three properties matter more than hit rate:

  correct     A cached answer must be the answer THIS surface would have
              produced. The key includes the namespace, so the ClawPump bot
              can never serve an answer the Bullpen concierge generated for
              the same words.
  fresh       Documentation changes. Entries expire, and a re-ingest can
              clear the cache outright rather than waiting them out.
  bounded     Memory is bounded by entry count, evicting least-recently-used.
              An unbounded cache on a public endpoint is a way for anyone to
              exhaust the process by asking many distinct questions.

Deliberately per-process and in-memory: this runs as a single instance, and
a shared cache would mean another dependency to operate and another thing
that can serve a stale answer after a deploy. If it ever runs multiple
instances, the worst case is each warming its own copy — not incorrectness.

NOT cached: any request carrying conversation history. A follow-up question
("what about for perps?") only means anything next to the turns before it,
so its text is not a key.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from typing import Any

# Whitespace and casing should not produce distinct entries; the words
# should. Trailing punctuation goes too, so "what are the fees" and "what
# are the fees?" are one entry rather than two.
_PUNCT = re.compile(r"[\s\?\!\.\,]+$")
_SPACE = re.compile(r"\s+")


def normalise(question: str) -> str:
    return _PUNCT.sub("", _SPACE.sub(" ", question.strip().lower()))


def make_key(question: str, *, surface: str, brief: bool = False,
             top_k: int | None = None) -> str:
    """Identity of a question, as asked on one surface.

    `surface` separates knowledge bases. Two bots answering "what are the
    fees" from different documentation must never collide, and a key that
    is only the question text would do exactly that.
    """
    raw = f"{surface}\x00{int(brief)}\x00{top_k or 0}\x00{normalise(question)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class AnswerCache:
    """Bounded, expiring, thread-safe store of finished answers."""

    def __init__(self, max_entries: int = 500, ttl_seconds: float = 86_400.0):
        self._max = max(0, max_entries)
        self._ttl = ttl_seconds
        # Guarded by a lock because entries are written from request handlers
        # running in different threads via the event loop's executor, and an
        # OrderedDict is not safe to mutate concurrently.
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._max > 0

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if now - stored_at > self._ttl:
                # Expired. Drop it rather than leaving it to be evicted
                # later, so a stale answer cannot be served even once.
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)      # least recently used

    def clear(self) -> int:
        """Drop everything. Call after re-ingesting, so a corrected document
        takes effect immediately instead of after the TTL."""
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def state(self) -> dict:
        with self._lock:
            size = len(self._data)
        total = self.hits + self.misses
        return {
            "entries": size,
            "max_entries": self._max,
            "ttl_seconds": int(self._ttl),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
