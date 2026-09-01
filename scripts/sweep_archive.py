"""Ask the archive its own questions before anybody else does.

    .venv/bin/python scripts/sweep_archive.py                 # everything indexed
    .venv/bin/python scripts/sweep_archive.py --count 200
    .venv/bin/python scripts/sweep_archive.py --episode x-2075316750439338088

Every question here is written FROM a passage, so the answer is known to
exist and the passage that holds it is known by name. That is the whole
point. A question invented from a topic list cannot tell a retrieval bug
from an honest gap -- when the bot says "I couldn't find that", there is
no way to know whether it failed or whether it was right. Here there is:
the passage is sitting there, so a miss is a bug, every time.

Two ways to fail, and the second is the one that gets screenshotted:

    the bot says it cannot find something that is definitely there. This
    looks broken, and it is the failure people report.

    the bot answers confidently from the wrong passage. This looks fine.
    Nobody reports it, and it is worse, because the reply goes out under
    a real person's name with a timestamp attached to it.

Both are counted. A question can retrieve the right passage and still be
refused, and it can be answered fluently having retrieved nothing of the
kind, so neither check substitutes for the other.

That distinction has already been got wrong three times on the ABSENT
list in ab_questions.py -- "Mount Gox", "Frentech", "om fork" and
"gonna cut instead of hiking" were all judged absent because the words
were absent, while the show discussed every one of them. Absence of the
words is not absence of the subject. Deriving the question from the
passage sidesteps that entirely.

Two stages, because the second one is where the money is:

    retrieve   every question, and check whether the source passage came
               back. No answer is generated, so this is embedding +
               Pinecone + rerank and nothing else.

    answer     only the questions that failed, so a human can read what a
               real asker would have seen. Failures are a small fraction,
               so the expensive call runs a small number of times.

This is coverage, and it is not the same job as ab_questions.py. That set
is fixed and hand-mined and answers "did this code change break what used
to work". This one is drawn fresh from the whole archive every run and
answers "is there anything in here nobody can reach". A change can pass
the first and fail this.

Sampled across every indexed episode by default, not one. Roughly 2,700
passages are indexed, so a couple of hundred questions a day walks the
whole archive in a fortnight, and a different sample each run means the
coverage compounds instead of re-asking the same ground.

Reads the same episodes.json the ingester reads and rebuilds windows with
the same settings, so a passage here is the passage in Pinecone rather
than something close to it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings          # noqa: E402
from app.podcast import PodcastIndex, _windows  # noqa: E402
from app.schemas import Episode              # noqa: E402
from app.x_bot import is_a_deflection, is_a_miss  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
LOGS = ROOT / "data" / "sweeps"

# A hit is the source passage if it is the same episode and starts within
# this many seconds. Not zero: windows overlap by two segments, so the
# passage carrying an answer often begins a few seconds either side of the
# one the question was written from, and calling that a miss would report
# failures that are not failures.
SAME_PASSAGE_SECONDS = 45.0

# Wider: right episode, wrong moment. Worth separating from a clean miss
# because the failure is different -- retrieval found the show but not the
# spot, which is a ranking problem rather than an invisibility problem.
SAME_NEIGHBOURHOOD_SECONDS = 240.0

_ASK = """Here is a passage from a podcast about crypto and investing.

Write ONE question that this passage answers, as a real listener would type
it to a search box. Rules:

- It must be answerable from THIS passage alone.
- Ask about the substance, never about the passage: no "in this clip",
  no "according to the transcript", no "what does the speaker say".
- Use the words a listener would use, not the transcript's words. If the
  passage says "Salana" the question says "Solana".
- Lowercase, no trailing question mark, under 15 words.
- If the passage is filler -- an ad read, crosstalk, a sign-off, nothing
  anybody would search for -- reply with exactly: SKIP

Passage:
{passage}

Question:"""


def _stamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


async def _write_question(client, model: str, passage: str) -> str | None:
    """One question for one passage, or None for a passage not worth asking."""
    try:
        reply = await client.messages.create(
            model=model,
            max_tokens=60,
            messages=[{"role": "user",
                       "content": _ASK.format(passage=passage[:2400])}],
        )
    except Exception as exc:                                   # noqa: BLE001
        print(f"    question generation failed ({exc})")
        return None
    text = "".join(b.text for b in reply.content
                   if getattr(b, "type", "") == "text").strip()
    text = text.strip().strip('"').rstrip("?").strip()
    if not text or text.upper().startswith("SKIP") or len(text) < 12:
        return None
    return text


def _classify(hits, episode_id: str, start: float) -> str:
    """Did retrieval put the source passage in front of the model?"""
    best = None
    for hit in hits:
        if hit.episode_id != episode_id:
            continue
        gap = abs(float(hit.start_seconds) - start)
        best = gap if best is None else min(best, gap)
    if best is None:
        return "MISS"
    if best <= SAME_PASSAGE_SECONDS:
        return "HIT"
    if best <= SAME_NEIGHBOURHOOD_SECONDS:
        return "NEAR"
    return "MISS"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", help="one episode id (default: all indexed)")
    ap.add_argument("--count", type=int, default=120,
                    help="questions to ask (default 120)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the passage sample, to repeat a run exactly")
    args = ap.parse_args()

    if not EPISODES.exists():
        print(f"  no {EPISODES} — run this where the episode data is")
        return 1
    episodes = json.loads(EPISODES.read_text())
    episodes.sort(key=lambda e: e.get("published_at") or "")
    if args.episode:
        episodes = [e for e in episodes if e["episode_id"] == args.episode]
        if not episodes:
            print(f"  no episode {args.episode}")
            return 1

    settings = get_settings()
    index = PodcastIndex(settings)

    # Every passage in everything indexed, rebuilt with the settings the
    # ingester used so that a passage here is the passage in Pinecone
    # rather than something near it.
    pool: list[tuple[str, str, float, str]] = []
    for episode in episodes:
        parsed = Episode(**episode)
        for start_t, text, _stamped in _windows(
                parsed.segments, settings.chunk_max_chars, overlap_segments=2):
            pool.append((episode["episode_id"], episode.get("title", ""),
                         start_t, text))
    if not pool:
        print("  no passages")
        return 1

    print(f"  {len(episodes)} episode(s)  ·  {len(pool)} passages indexed")

    # A fresh sample every run, so a fortnight of these walks the whole
    # archive instead of re-asking the same ground each day.
    rng = random.Random(args.seed)
    picked = pool if args.count >= len(pool) else rng.sample(pool, args.count)
    picked.sort(key=lambda w: (w[0], w[2]))
    print(f"  asking {len(picked)}\n")

    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    # The same small model the search itself uses. Writing a
    # question from a passage in front of it is easier than
    # answering one, so nothing bigger is warranted.
    model = settings.search_model

    gate = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []
    began = time.time()

    async def one(episode_id: str, title: str, start: float,
                  text: str) -> None:
        async with gate:
            question = await _write_question(client, model, text)
            if not question:
                return
            try:
                hits = await index.retrieve(question)
            except Exception as exc:                           # noqa: BLE001
                rows.append({"question": question, "verdict": "ERROR",
                             "error": str(exc), "at": _stamp(start),
                             "start_seconds": start, "episode_id": episode_id})
                print(f"  ERROR  {question}\n         {exc}")
                return
            verdict = _classify(hits, episode_id, start)
            rows.append({
                "question": question, "verdict": verdict,
                "at": _stamp(start), "start_seconds": start,
                "episode_id": episode_id, "title": title,
                "returned": [
                    {"episode_id": h.episode_id, "at": h.timestamp}
                    for h in hits[:6]],
            })
            if verdict != "HIT":
                print(f"  {verdict:5}  {question}")
                print(f"         should be {episode_id} {_stamp(start)}")

    await asyncio.gather(*(one(e, ti, s, tx) for e, ti, s, tx in picked))

    # An answer is generated for every question that failed retrieval, and
    # for a sample of the ones that passed. Both are needed: retrieval can
    # succeed while the model still declines, and that refusal is exactly
    # what somebody screenshots. Capping the passing sample is what keeps
    # this cheap -- the expensive call is the answer, not the lookup.
    failed = [r for r in rows if r["verdict"] in ("MISS", "NEAR")]
    passed = [r for r in rows if r["verdict"] == "HIT"]
    checked = failed + rng.sample(passed, min(len(passed), max(20, args.count // 4)))
    if checked:
        print(f"\n  answering {len(checked)} ({len(failed)} that failed "
              f"retrieval, {len(checked) - len(failed)} that passed) to see "
              f"what an asker would have got\n")

        async def answer(row: dict) -> None:
            async with gate:
                try:
                    result = await index.search(row["question"])
                except Exception as exc:                       # noqa: BLE001
                    row["answer"] = f"(search failed: {exc})"
                    row["refused"] = None
                    return
                row["answer"] = result.answer
                row["refused"] = bool(
                    is_a_miss(result.answer) or is_a_deflection(result.answer))
                if row["refused"] and row["verdict"] == "HIT":
                    # The worst kind: the passage was right there in front
                    # of the model and it still said it could not find it.
                    print(f"  REFUSED (but retrieval was fine)  "
                          f"{row['question']}")
                    print(f"         {row['answer'][:150]}")

        await asyncio.gather(*(answer(r) for r in checked))

    counts = {v: sum(1 for r in rows if r["verdict"] == v)
              for v in ("HIT", "NEAR", "MISS", "ERROR")}
    asked = len(rows)
    refused = [r for r in rows if r.get("refused")]
    refused_despite = [r for r in refused if r["verdict"] == "HIT"]
    print("\n" + "=" * 68)
    print(f"  asked {asked} in {time.time() - began:.0f}s")
    print("\n  retrieval — did the source passage come back?")
    for name in ("HIT", "NEAR", "MISS", "ERROR"):
        n = counts[name]
        if asked:
            print(f"    {name:6} {n:4}  {100 * n / asked:5.1f}%")
    if checked:
        print(f"\n  answers — of {len(checked)} generated:")
        print(f"    refused                    {len(refused):4}")
        print(f"      of those, retrieval was fine {len(refused_despite):4}"
              f"   <- prompt problem, not a lookup problem")
    print("=" * 68)

    LOGS.mkdir(parents=True, exist_ok=True)
    scope = args.episode or "archive"
    out = LOGS / f"{scope}-{int(began)}.jsonl"
    with out.open("w") as fh:
        for row in sorted(rows, key=lambda r: r["start_seconds"]):
            fh.write(json.dumps(row) + "\n")
    print(f"\n  log: {out}")
    if counts["MISS"]:
        print(f"  {counts['MISS']} passage(s) cannot be found by a question "
              f"written from them — grep the log for MISS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
