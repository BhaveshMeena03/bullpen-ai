"""Run the question corpus against a live concierge and record every answer.

    python evals/run_eval.py [--base URL] [--limit N] [--out PATH]

Writes one JSON object per question to evals/results.jsonl as it goes, so
a run that dies partway keeps everything it already paid for, and a rerun
skips what's already there.

Paced under the service's own per-IP rate limit. The point is to measure
the concierge, not to trip its throttle and measure that instead.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.questions import corpus  # noqa: E402

DEFAULT_BASE = "https://concierge.lexthedev.com"
OUT = Path(__file__).resolve().parent / "results.jsonl"

# RATE_LIMIT_RPM defaults to 30/min per IP. 2.5s between calls leaves
# headroom for a retry without ever approaching the ceiling.
DELAY_SECONDS = 2.5

# Phrases the concierge uses when it can't ground an answer. Kept in sync
# with _UNKNOWN_MARKERS in app/main.py — the eval classifies from the
# outside, so it can't import the server's copy.
UNKNOWN_MARKERS = (
    "don't have that information",
    "do not have that information",
    "don't have specific",
    "don't have information on",
    "couldn't find that",
    "could not find that",
    "i don't have details",
    "contact official bullpen support",
    "official bullpen support channels",
    "reach out to official",
)


def classify(row: dict, answer: str, sources: list, refused: bool) -> str:
    """answered | no_context | low_confidence | refused"""
    if refused:
        return "refused"
    if not sources:
        return "no_context"
    if any(m in answer.lower() for m in UNKNOWN_MARKERS):
        return "low_confidence"
    return "answered"
    # NOTE: this is a rough live-progress label only. report.py re-derives
    # every outcome from the stored answer via evals/classify.py, which is
    # the authority — this copy cannot see refusals stated in prose.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = corpus()
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["q"])
    todo = [r for r in rows if r["q"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(rows)} questions, {len(done)} already done, {len(todo)} to run")
    if not todo:
        return 0
    print(f"~{len(todo) * DELAY_SECONDS / 60:.1f} min at {DELAY_SECONDS}s spacing\n")

    failures = 0
    with args.out.open("a") as fh, httpx.Client(timeout=120.0) as client:
        for i, row in enumerate(todo, 1):
            try:
                r = client.post(
                    f"{args.base}/v1/chat",
                    json={"message": row["q"], "history": []},
                )
                r.raise_for_status()
                d = r.json()
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  [{i}/{len(todo)}] ERROR {row['q'][:44]}: {exc}")
                time.sleep(DELAY_SECONDS)
                continue

            answer = d.get("answer", "")
            sources = d.get("sources", []) or []
            outcome = classify(row, answer, sources, d.get("refused", False))
            fh.write(json.dumps({
                **row,
                "outcome": outcome,
                "answer": answer,
                "n_sources": len(sources),
                "source_titles": [s.get("title", "") for s in sources[:3]],
            }, ensure_ascii=False) + "\n")
            fh.flush()

            mark = {"answered": "  ", "no_context": "!!", "low_confidence": " ?",
                    "refused": "><"}[outcome]
            print(f"  [{i}/{len(todo)}] {mark} {outcome:<15} {row['q'][:52]}")
            time.sleep(DELAY_SECONDS)

    print(f"\nwrote {args.out}  ({failures} request failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
