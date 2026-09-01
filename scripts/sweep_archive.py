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
it to a search box, and say which kind of question it is.

Rules for the question:
- It must be answerable from THIS passage alone.
- Ask about the substance, never about the passage or the people saying
  it: no "in this clip", no "according to the transcript", no "the
  speaker", no "the host", no "the guest". A listener asking a search box
  does not know who was talking.
- Use the words a listener would use, not the transcript's words. If the
  passage says "Salana" the question says "Solana".
- Lowercase, no trailing question mark, under 15 words.

Then label it:
- SPECIFIC if this passage is essentially the only place in a crypto
  podcast archive that could answer it -- it names a person, company,
  token, number, date or one particular event. "what did luca netz pay
  for pudgy penguins" is SPECIFIC.
- GENERAL if a dozen other passages across many episodes could answer it
  just as well, because it asks about a recurring theme rather than a
  particular moment. "why do most crypto projects fail" is GENERAL.

Reply on one line as either
    SPECIFIC: <question>
or  GENERAL: <question>

If the passage is filler -- an ad read, crosstalk, a sign-off, nothing
anybody would search for -- reply with exactly: SKIP

Passage:
{passage}

Answer:"""


def _stamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


async def _write_question(client, model: str,
                          passage: str) -> tuple[str, str] | None:
    """(kind, question) for one passage, or None for one not worth asking.

    `kind` decides how the answer is judged, so it matters as much as the
    question. A GENERAL question that comes back with a different passage
    has not failed -- a dozen passages answer it equally well, and picking
    one of them is the correct behaviour rather than a miss.
    """
    try:
        reply = await client.messages.create(
            model=model,
            max_tokens=80,
            messages=[{"role": "user",
                       "content": _ASK.format(passage=passage[:2400])}],
        )
    except Exception as exc:                                   # noqa: BLE001
        print(f"    question generation failed ({exc})")
        return None
    text = "".join(b.text for b in reply.content
                   if getattr(b, "type", "") == "text").strip()
    if text.upper().startswith("SKIP"):
        return None

    kind = "GENERAL"
    for label in ("SPECIFIC", "GENERAL"):
        if text.upper().startswith(label):
            kind, text = label, text[len(label):].lstrip(":").strip()
            break
    text = text.strip().strip('"').rstrip("?").strip()
    if not text or len(text) < 12:
        return None
    # A question naming the show rather than the subject was never a fair
    # test: nobody types "what does the speaker say" into a search box.
    if any(bad in text.lower() for bad in
           ("the speaker", "the host", "the guest", "this clip",
            "the passage", "the transcript", "the excerpt")):
        return None
    return kind, text


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
            written = await _write_question(client, model, text)
            if not written:
                return
            kind, question = written
            try:
                hits = await index.retrieve(question)
            except Exception as exc:                           # noqa: BLE001
                rows.append({"question": question, "kind": kind,
                             "verdict": "ERROR",
                             "error": str(exc), "at": _stamp(start),
                             "start_seconds": start, "episode_id": episode_id})
                print(f"  ERROR  {question}\n         {exc}")
                return
            found = _classify(hits, episode_id, start)
            # Only a SPECIFIC question can fail by returning the wrong
            # passage. For a GENERAL one a different passage is a different
            # correct answer, and calling that a miss reported failures
            # that were not failures -- eight of ten on the first run.
            # What a GENERAL question can still fail is being refused, and
            # that is decided further down, once there is an answer to read.
            verdict = found if kind == "SPECIFIC" else "OPEN"
            rows.append({
                "question": question, "kind": kind, "verdict": verdict,
                "found": found,
                "at": _stamp(start), "start_seconds": start,
                "episode_id": episode_id, "title": title,
                "returned": [
                    {"episode_id": h.episode_id, "at": h.timestamp}
                    for h in hits[:6]],
            })
            if verdict in ("MISS", "NEAR"):
                print(f"  {verdict:5}  {question}")
                print(f"         should be {episode_id} {_stamp(start)}")

    await asyncio.gather(*(one(e, ti, s, tx) for e, ti, s, tx in picked))

    # An answer is generated for every question that failed retrieval, and
    # for a sample of the ones that passed. Both are needed: retrieval can
    # succeed while the model still declines, and that refusal is exactly
    # what somebody screenshots. Capping the passing sample is what keeps
    # this cheap -- the expensive call is the answer, not the lookup.
    failed = [r for r in rows if r["verdict"] in ("MISS", "NEAR")]
    # Every GENERAL question needs an answer, because being refused is the
    # only way it can fail and that cannot be seen without one.
    general = [r for r in rows if r["verdict"] == "OPEN"]
    passed = [r for r in rows if r["verdict"] == "HIT"]
    checked = failed + general + rng.sample(
        passed, min(len(passed), max(20, args.count // 4)))
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
                if row["refused"]:
                    # For a GENERAL question this IS the failure: the show
                    # covers the topic in many places and the bot said it
                    # could not find it. For a SPECIFIC one that retrieved
                    # correctly it is worse -- the passage was in front of
                    # the model and it still declined.
                    where = ("retrieval was fine" if row["found"] == "HIT"
                             else f"retrieval {row['found'].lower()}")
                    print(f"  REFUSED [{row['kind']}, {where}]  "
                          f"{row['question']}")
                    print(f"         {row['answer'][:150]}")

        await asyncio.gather(*(answer(r) for r in checked))

    asked = len(rows)
    specific = [r for r in rows if r["kind"] == "SPECIFIC"]
    generals = [r for r in rows if r["kind"] == "GENERAL"]
    refused = [r for r in rows if r.get("refused")]

    print("\n" + "=" * 68)
    print(f"  asked {asked} in {time.time() - began:.0f}s")

    if specific:
        print(f"\n  SPECIFIC ({len(specific)}) — one passage answers it, so "
              f"the passage has to come back")
        for name in ("HIT", "NEAR", "MISS", "ERROR"):
            n = sum(1 for r in specific if r["verdict"] == name)
            print(f"    {name:6} {n:4}  {100 * n / len(specific):5.1f}%")

    if generals:
        # A different passage is a different right answer here, so the only
        # failure is a refusal on a subject the show returns to constantly.
        bad = sum(1 for r in generals if r.get("refused"))
        print(f"\n  GENERAL ({len(generals)}) — many passages answer it, so "
              f"only a refusal is a failure")
        print(f"    answered {len(generals) - bad:4}  "
              f"{100 * (len(generals) - bad) / len(generals):5.1f}%")
        print(f"    refused  {bad:4}  {100 * bad / len(generals):5.1f}%"
              f"   <- the show covers this and we said we could not find it")

    hard = [r for r in refused if r.get("found") == "HIT"]
    if hard:
        print(f"\n  {len(hard)} refused with the right passage already "
              f"retrieved — a prompt problem, not a lookup problem")
    print("=" * 68)

    LOGS.mkdir(parents=True, exist_ok=True)
    scope = args.episode or "archive"
    out = LOGS / f"{scope}-{int(began)}.jsonl"
    with out.open("w") as fh:
        for row in sorted(rows, key=lambda r: r["start_seconds"]):
            fh.write(json.dumps(row) + "\n")
    print(f"\n  log: {out}")
    unreachable = sum(1 for r in specific if r["verdict"] == "MISS")
    if unreachable:
        print(f"  {unreachable} passage(s) cannot be found by a question "
              f"written from them — grep the log for MISS")
    if refused:
        print(f"  {len(refused)} refusal(s) — grep the log for refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
