"""Time to first token, proxied against direct, on a real payload.

    env -u ANTHROPIC_BASE_URL .venv/bin/python scripts/bench_inference.py
    env -u ANTHROPIC_BASE_URL .venv/bin/python scripts/bench_inference.py --runs 5

Answers one question: is UsePod slower than Anthropic right now, and by
how much. It exists because "it feels slow" is not something to send a
supplier, and because the first version of this measurement was wrong in
a way worth not repeating -- it timed a twenty-token prompt against a
production call that carries six retrieved passages, so it compared a
postcard with a parcel and the gap it found meant nothing.

So the payload here is built from the archive: six real passages at the
real window size, which is what every answer actually sends.

Three rules this obeys, none of them optional:

  * The real Anthropic key NEVER goes to the proxy. Routing is decided by
    app.config.anthropic_client_kwargs -- the same function the service
    uses -- which compares the PARSED HOSTNAME against api.anthropic.com
    exactly, because "api.anthropic.com" is a substring of
    api.anthropic.com.evil.example and a substring test would hand the key
    to it.
  * The proxy token is a PATH segment, so it appears in exception text,
    stack traces and httpx's request repr. Everything printed goes through
    redact() first, and redaction happens BEFORE truncation -- cutting a
    URL to eighty characters can leave the token and drop the context that
    made it recognisable.
  * ANTHROPIC_BASE_URL is exported in the operator's shell and overrides
    .env, so this must run under `env -u ANTHROPIC_BASE_URL` or it will
    silently measure direct twice and report no difference.

Streaming, and it stops at the first text token: that is the number a
person waiting on an answer actually feels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import (anthropic_client_kwargs,  # noqa: E402
                        get_settings, redact)

EPISODES = ROOT / "data" / "episodes.json"

# What a real answer carries: retrieval_top_k passages at chunk_max_chars.
PASSAGES = 6
PASSAGE_CHARS = 2400

QUESTION = "what did they say about zcash and why it might outperform"


def passages() -> list[str]:
    """Six windows of real transcript, the size retrieval actually sends."""
    episodes = json.loads(EPISODES.read_text())
    longest = max(episodes, key=lambda e: len(e.get("segments") or []))
    text = " ".join(s.get("text", "") for s in longest["segments"])
    step = max(1, len(text) // (PASSAGES + 1))
    out = []
    for i in range(PASSAGES):
        start = i * step
        out.append(text[start:start + PASSAGE_CHARS])
    return out


def prompt() -> str:
    body = "\n\n".join(f"[passage {i + 1}]\n{p}"
                       for i, p in enumerate(passages()))
    return (f"{body}\n\nUsing only the passages above, answer: {QUESTION}\n"
            "Quote what was said and give the timestamp.")


async def first_token(client: AsyncAnthropic, model: str,
                      text: str) -> float | None:
    """Seconds until the first text token. None if the call failed."""
    began = time.perf_counter()
    try:
        async with client.messages.stream(
            model=model, max_tokens=300,
            messages=[{"role": "user", "content": text}],
        ) as stream:
            async for chunk in stream.text_stream:
                if chunk:
                    return time.perf_counter() - began
    except Exception as exc:                                   # noqa: BLE001
        # redact BEFORE truncating; the token is in the path.
        print(f"      failed: {redact(str(exc))[:200]}")
        return None
    return None


async def run(label: str, client: AsyncAnthropic, model: str,
              text: str, runs: int) -> list[float]:
    print(f"  {label}")
    times: list[float] = []
    for i in range(runs):
        taken = await first_token(client, model, text)
        if taken is None:
            continue
        times.append(taken)
        print(f"      run {i + 1}: {taken:.2f}s")
    return times


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    settings = get_settings()
    kwargs = anthropic_client_kwargs(settings)
    if "base_url" not in kwargs:
        print("  ANTHROPIC_BASE_URL is not set, so there is no proxy to "
              "compare against.\n  Run under `env -u ANTHROPIC_BASE_URL` "
              "so .env is what decides.")
        return 2

    model = settings.search_model or settings.anthropic_model
    text = prompt()
    print(f"\n  model {model} · {len(text):,} chars of payload "
          f"(~{len(text) // 4:,} tokens) · {PASSAGES} passages")
    print(f"  proxy {redact(kwargs['base_url'])}\n")

    proxied = AsyncAnthropic(**kwargs)
    direct = AsyncAnthropic(api_key=settings.anthropic_api_key)

    through = await run("through usepod", proxied, model, text, args.runs)
    print()
    straight = await run("anthropic direct", direct, model, text, args.runs)

    print()
    if through and straight:
        a, b = statistics.median(through), statistics.median(straight)
        print(f"  median   usepod {a:.2f}s   direct {b:.2f}s   "
              f"({a / b:.1f}x)" if b else "")
    elif straight and not through:
        print("  usepod returned nothing; direct worked. That is an "
              "outage, not a latency measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
