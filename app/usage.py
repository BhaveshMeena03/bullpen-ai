"""What the models actually cost, per day and per surface.

Anthropic has no API for reading an account balance, so this cannot mirror
the console. What it can do is count exactly what THIS service spent, from
the usage block on every response, and price it from the published rates.
That is an estimate, and it will read slightly low if a request was retried
upstream or served by a fallback model — but it is derived from real token
counts rather than guessed from request counts, so it is close.

The number that matters here is not the total. It is the split: which
surface, which model, and how much the cache saved, because those are the
three things you can actually change.

Durability: the ledger is written to disk so a restart does not lose the
day. It is NOT a database — on a host with an ephemeral filesystem a
redeploy still clears it, which is why every entry is also emitted as an
ANALYTICS log line. The logs are the durable record; this file is the fast
one.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def writable_path(preferred: Path) -> Path:
    """`preferred` if its directory takes a write, else the temp directory.

    The image ships at /srv with a read-only data directory, so the obvious
    path raises PermissionError on every single save — once per answered
    question, each one a warning line. The ledger survives that (it keeps
    counting in memory, and the ANALYTICS lines are the durable record), but
    the noise buries real warnings in the log.

    Same probe the bot state uses, for the same reason. Losing the file is
    already expected on an ephemeral disk; being unable to write it at all
    is not worth a warning per request.
    """
    try:
        preferred.parent.mkdir(parents=True, exist_ok=True)
        probe = preferred.parent / ".write-test"
        probe.touch()
        probe.unlink()
        return preferred
    except OSError:
        return Path(tempfile.gettempdir()) / preferred.name


# USD per million tokens, from Anthropic's published pricing. Cache reads are
# 0.1x the input rate; cache writes are 1.25x.
#
# Prefix-matched, longest first, so a dated model id ("claude-haiku-4-5-2025…")
# still prices correctly rather than silently falling through to a default.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
_FALLBACK = (3.00, 15.00)

CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


def _canonical(model: str) -> str:
    """The name this table knows, from whatever the API called it back.

    A proxy answers with its own naming. usepod returns
    "anthropic/claude-haiku-4.5" where Anthropic returns
    "claude-haiku-4-5-20251001" — a vendor prefix, and dots where this
    table has dashes. Neither matched, so every proxied call was priced at
    the mid-tier fallback of $3/$15 against Haiku's actual $1/$5, and the
    spend ledger read about seven times the real number. A cost table that
    is wrong in the expensive direction is worse than no cost table: it
    argues against the cheaper route on made-up evidence.
    """
    name = (model or "").strip().lower()
    if "/" in name:                       # vendor prefix, e.g. "anthropic/"
        name = name.rsplit("/", 1)[-1]
    # "claude-haiku-4.5" -> "claude-haiku-4-5", leaving date suffixes alone.
    return re.sub(r"(\d)\.(\d)", r"\1-\2", name)


def price_for(model: str) -> tuple[float, float]:
    name = _canonical(model)
    for prefix in sorted(PRICES, key=len, reverse=True):
        if name.startswith(prefix):
            return PRICES[prefix]
    # An unknown model is priced at the mid tier rather than zero. Zero would
    # make a new model look free, which is the failure that matters here.
    logger.warning("no price for model %r — using default", model)
    return _FALLBACK


def cost_of(model: str, *, input_tokens: int = 0, output_tokens: int = 0,
            cache_read: int = 0, cache_write: int = 0) -> float:
    rate_in, rate_out = price_for(model)
    return (
        input_tokens * rate_in
        + output_tokens * rate_out
        + cache_read * rate_in * CACHE_READ_MULTIPLIER
        + cache_write * rate_in * CACHE_WRITE_MULTIPLIER
    ) / 1_000_000


class UsageLedger:
    """Per-day, per-surface, per-model token and cost totals."""

    def __init__(self, path: Path | None = None, keep_days: int = 60):
        self._path = path
        self._keep = keep_days
        self._lock = threading.Lock()
        self._days: dict[str, dict] = {}
        self._load()

    # --- recording ---------------------------------------------------------

    def record(self, surface: str, model: str, usage: dict | None,
               *, cached: bool = False) -> None:
        """Add one model call. `cached: True` records a cache hit — no tokens
        were spent, but the saving is worth counting, because "the cache saved
        you $X" is the only way to know whether it is earning its complexity.
        """
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        with self._lock:
            bucket = self._days.setdefault(day, {"surfaces": {}, "saved_usd": 0.0})
            s = bucket["surfaces"].setdefault(surface, {})
            m = s.setdefault(model, {"calls": 0, "input": 0, "output": 0,
                                     "cache_read": 0, "cache_write": 0,
                                     "usd": 0.0, "cache_hits": 0})
            if cached:
                m["cache_hits"] += 1
                # What this call WOULD have cost, using this surface's own
                # average so far. Without a prior sample there is nothing
                # honest to claim, so it counts as zero rather than a guess.
                avg = m["usd"] / m["calls"] if m["calls"] else 0.0
                bucket["saved_usd"] += avg
            else:
                u = usage or {}
                inp = int(u.get("input_tokens") or 0)
                out = int(u.get("output_tokens") or 0)
                cr = int(u.get("cache_read_input_tokens") or 0)
                cw = int(u.get("cache_creation_input_tokens") or 0)
                m["calls"] += 1
                m["input"] += inp
                m["output"] += out
                m["cache_read"] += cr
                m["cache_write"] += cw
                m["usd"] += cost_of(model, input_tokens=inp, output_tokens=out,
                                    cache_read=cr, cache_write=cw)
            self._prune()
        logger.info("ANALYTICS %s", json.dumps({
            "event": "usage", "day": day, "surface": surface,
            "model": model, "cached": cached, **(usage or {}),
        }))
        self._save()

    # --- reading -----------------------------------------------------------

    def report(self) -> dict:
        with self._lock:
            days = json.loads(json.dumps(self._days))     # deep copy
        out_days = []
        for day in sorted(days, reverse=True):
            b = days[day]
            total = 0.0
            calls = 0
            hits = 0
            surfaces = {}
            for surface, models in b["surfaces"].items():
                s_usd = sum(m["usd"] for m in models.values())
                s_calls = sum(m["calls"] for m in models.values())
                s_hits = sum(m["cache_hits"] for m in models.values())
                total += s_usd
                calls += s_calls
                hits += s_hits
                surfaces[surface] = {
                    "usd": round(s_usd, 4), "calls": s_calls,
                    "cache_hits": s_hits,
                    "models": {k: {**v, "usd": round(v["usd"], 4)}
                               for k, v in models.items()},
                }
            out_days.append({
                "day": day, "usd": round(total, 4), "calls": calls,
                "cache_hits": hits,
                "saved_usd": round(b.get("saved_usd", 0.0), 4),
                "surfaces": surfaces,
            })

        month = sum(d["usd"] for d in out_days[:30])
        recent = [d for d in out_days[:7] if d["calls"] or d["cache_hits"]]
        per_day = (sum(d["usd"] for d in recent) / len(recent)) if recent else 0.0
        # Runway from a handful of calls is not an estimate, it is noise
        # dressed as a number — three test requests extrapolated to "$100
        # lasts 2433 days". Withheld until there is enough traffic for the
        # average to mean anything.
        sample = sum(d["calls"] for d in recent)
        return {
            "days": out_days,
            "total_usd": round(sum(d["usd"] for d in out_days), 4),
            "last_30d_usd": round(month, 4),
            "saved_usd": round(sum(d["saved_usd"] for d in out_days), 4),
            "avg_usd_per_active_day": round(per_day, 4),
            # Plain-language runway, which is the question actually being
            # asked. Meaningless without a few days of data, so it is null
            # until there are some.
            "days_per_100_usd": (round(100 / per_day, 1)
                                 if per_day > 0.0001 and sample >= 25 else None),
            "runway_sample_calls": sample,
        }

    # --- persistence -------------------------------------------------------

    def _prune(self) -> None:
        if len(self._days) <= self._keep:
            return
        for day in sorted(self._days)[:-self._keep]:
            del self._days[day]

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            self._days = json.loads(self._path.read_text())
        except Exception as exc:      # noqa: BLE001
            # A corrupt ledger must never stop the service booting; the worst
            # case is losing accounting history, not answering questions.
            logger.warning("could not read usage ledger: %s", exc)
            self._days = {}

    def _save(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with self._lock:
                tmp.write_text(json.dumps(self._days))
            tmp.replace(self._path)   # atomic, so a crash mid-write cannot
                                      # leave a truncated file behind
        except Exception as exc:      # noqa: BLE001
            logger.warning("could not write usage ledger: %s", exc)
