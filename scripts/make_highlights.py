"""Build a pool of striking moments the bot can offer unprompted.

    .venv/bin/python scripts/make_highlights.py
    .venv/bin/python scripts/make_highlights.py --per-episode 3

People tag the account to say "very cool" far more often than to ask it
anything. Silence is the safe answer and a wasted moment: someone is looking
at the account, and the one thing that would convince them is a demonstration
rather than a thank-you.

So each compliment gets a different genuinely interesting moment from the
archive. Generated once, here, rather than per reply — a model call at reply
time costs money, adds latency, and can produce a dud in public. Picking from
a pool that was reviewed beforehand cannot.

Written to data/highlights.json. Costs roughly $0.008 per episode, so about
$0.25 for the whole catalogue, and only needs rerunning when episodes are
added.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.podcast import _deep_link, _timestamp  # noqa: E402
from app.x_bot import _seconds  # noqa: E402

# The shapes the model uses when it cannot tell who was speaking.
_UNSURE = re.compile(
    r"""(?ix) unnamed\s+speaker | speaker\s+(?:identity|unclear)
      | unclear\s+from\s+(?:the\s+)?transcript | identity\s+unclear
      | (?:likely|possibly|presumably)\s+(?:said|stated|claimed)
      | \bunidentified\b""")

EPISODES = ROOT / "data" / "episodes.json"
OUT = ROOT / "data" / "highlights.json"

PROMPT = """\
Below is a transcript of one episode of a crypto podcast, with a timestamp \
on every line.

Pick the {n} most striking, concrete moments in it — the things someone \
would repeat to a friend. A specific number, a real decision, a claim that \
sounds wrong until you hear the reason. Not general discussion, not \
introductions, not "they talked about X".

For each, write ONE sentence naming WHO said it and what, followed by the \
timestamp of the line you took it from, exactly as it appears in brackets.

Two hard rules, because these get posted publicly with nobody checking them \
first:

1. Name the person. If the transcript does not make clear who is speaking, \
skip that moment entirely. "Someone claims" is not worth posting — it reads \
as a rumour and it is the archive's whole value to know who said what.

2. Quote numbers exactly as they were said. Never compute one. Do not turn \
"450 to 500" into a percentage, do not convert, do not total anything up. An \
arithmetic slip becomes a false claim under someone's name, and one already \
did: "$450 to $500 — roughly 50%", which is 11%.

Format, one per line, nothing else:
<sentence> | <timestamp>

If fewer than {n} moments in this transcript genuinely qualify, return fewer. \
A thin moment is worse than no moment.
"""


FUNNY_PROMPT = """\
Below is a transcript of one episode of a crypto podcast, with a timestamp \
on every line.

Pick the {n} funniest moments in it — the bits someone would clip and send \
to a friend. A line that lands, a story that goes somewhere stupid, a \
reaction, someone being roasted by their own admission.

For each, write ONE sentence naming WHO said or did it and what happened, \
followed by the timestamp of the line you took it from, exactly as it \
appears in brackets.

These get posted publicly, on their own, with nobody checking them first. \
So:

1. Name the person. If the transcript does not make clear who is speaking, \
skip that moment. An unattributed joke reads as a rumour.

2. The hosts and guests joking with each other IS the show, and quoting \
one of them is fine — if Ansem or Banks or a guest made the joke, it can be \
repeated. What is not fine is a joke aimed at somebody who is not there to \
take it: a person outside the conversation, mocked for how they look, their \
weight, their intelligence, or their relationships. The test is whether the \
person being joked about was in the room and part of it.

3. It has to be funny without the video. No "you had to see his face".

3a. Nothing about crime committed against someone, illness, addiction, or \
money someone lost. A robbery does not become a joke because it has an \
absurd detail in it, and the reply announces these as jokes — so the \
account would be the one calling it that.

3b. A joke someone makes at their own expense is always fine. So is one \
host ribbing another, or a guest, since they are all in the conversation \
and trading it back. A joke at the expense of someone outside it is not.

4. Quote numbers exactly as they were said. Never compute one.

Format, one per line, nothing else:
<sentence> | <timestamp>

If fewer than {n} moments genuinely qualify, return fewer. A weak joke \
posted unprompted is worse than no joke.
"""


async def highlights_for(client, model, episode: dict, n: int,
                         prompt: str | None = None,
                         kind: str = "fact") -> list[dict]:
    lines = [f"[{_timestamp(s['t'])}] {s['text'].strip()}"
             for s in episode["segments"] if s.get("text", "").strip()]
    # Every third line, capped: enough of the shape of the episode to judge
    # what stands out, without paying to send four hours of speech.
    sampled = "\n".join(lines[::3][:1400])

    response = await client.messages.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content":
                   f"{(prompt or PROMPT).format(n=n)}\n\n"
                   f"<transcript>\n{sampled}\n</transcript>"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")

    out = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        said, _, stamp = line.rpartition("|")
        said, stamp = said.strip(" -–—"), stamp.strip().strip("[]")
        # The model sometimes wraps its answer in the tag from the format
        # block. Four of these reached the live pool and would have posted
        # "<sentence>…</sentence>" to the timeline.
        said = re.sub(r"</?[a-z_]+>", "", said).strip()
        if len(said) < 40 or not stamp:
            continue
        # Rule 1 of the prompt is to name the person. Only genuinely
        # anonymous attributions are rejected — a one-letter name is a
        # nickname on this show, not a missing source: Banks calls Ansem Z.
        if re.match(r"^(?:someone|a guest|one host|a host)\b", said, re.I):
            continue
        # And the model hedging about who spoke, anywhere in the sentence.
        # One of these reached the pool complete with its own parenthetical:
        # "(unnamed speaker's estimate, though speaker identity unclear from
        # transcript)" — which is the admission that it broke rule 1, left
        # inside the thing that would have been posted.
        if _UNSURE.search(said):
            continue
        if not cites_its_moment(episode, stamp, said):
            print(f"     dropped (timestamp does not match): {said[:60]}")
            continue
        entry = _entry(episode, stamp, said)
        entry["kind"] = kind
        out.append(entry)
    return out


# How far either side of the cited moment to look for the claim. Wide
# enough for a sentence that runs on, narrow enough that a citation three
# hours out cannot pass.
VERIFY_WINDOW = 120


def _near(episode: dict, seconds: int) -> str:
    return " ".join(s.get("text", "") for s in episode["segments"]
                    if abs(s.get("t", 0) - seconds) <= VERIFY_WINDOW).lower()


def cites_its_moment(episode: dict, stamp: str, said: str) -> bool:
    """Does the transcript at that timestamp actually contain the claim?

    Nothing checked this, and two entries in the live pool were wrong. One
    said "Ansem said SpaceX traded at 175 on Hyperliquid" and pointed at
    3:07:00, where the show is running a bracket on who is best looking —
    the real discussion is at 14:07, nearly three hours earlier.

    That is the worst thing this archive can do. A wrong answer is a bad
    answer; a confident citation to a moment that says something else is
    the thing the whole tool exists to be trusted about.

    Deliberately shallow: it looks for the numbers and proper nouns in the
    sentence, not for meaning. A model call would judge better and would
    also be the same kind of judgement that produced the error.
    """
    try:
        window = _near(episode, _seconds(stamp))
    except ValueError:
        return False
    numbers = re.findall(r"\d[\d.,]*", said)
    names = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", said)
    if any(n.rstrip(".,") in window for n in numbers):
        return True
    return any(n.lower() in window for n in names)


def _entry(episode: dict, stamp: str, said: str) -> dict:
    """One pool entry, including where to watch the moment.

    The link is built here rather than at reply time so the pool can be
    read and checked before any of it is posted — same reason the facts
    are written ahead. _deep_link is the one the search path uses, so a
    highlight and an answer citing the same second produce the same URL.
    """
    try:
        seconds = _seconds(stamp)
    except ValueError:
        # A malformed timestamp costs the link, not the highlight.
        return {"episode_id": episode["episode_id"], "title": episode["title"],
                "timestamp": stamp, "text": said}
    return {
        "episode_id": episode["episode_id"],
        "title": episode["title"],
        "timestamp": stamp,
        "text": said,
        "url": _deep_link(episode["url"], episode.get("platform", ""), seconds),
    }


def relink(pool: list[dict], episodes: list[dict]) -> tuple[list[dict], int]:
    """Add the watch link to entries built before the pool carried one.

    Rebuilding the pool instead would cost a model call per episode and,
    worse, would replace facts that were read before they were trusted
    with new ones that nobody has looked at.
    """
    by_id = {e["episode_id"]: e for e in episodes}
    added = 0
    for entry in pool:
        if entry.get("url"):
            continue
        episode = by_id.get(entry.get("episode_id"))
        if not episode:
            continue
        built = _entry(episode, entry.get("timestamp", ""), entry.get("text", ""))
        if built.get("url"):
            entry["url"] = built["url"]
            added += 1
    return pool, added


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-episode", type=int, default=2)
    ap.add_argument("--only", action="append", default=[], metavar="EPISODE_ID",
                    help="generate for these episodes only and APPEND to the "
                         "pool. Without it the pool is rebuilt from scratch, "
                         "which throws away every hand-curated decision — "
                         "adding one broadcast that way took the pool from "
                         "15 entries to 6 and lost the reviewed jokes.")
    ap.add_argument("--kind", choices=("fact", "funny"), default="fact",
                    help="what to look for. 'funny' appends to the pool "
                         "rather than replacing it")
    ap.add_argument("--verify", action="store_true",
                    help="check the existing pool against the transcripts "
                         "and drop entries whose timestamp does not match")
    ap.add_argument("--relink", action="store_true",
                    help="only add missing watch links to the existing pool "
                         "— no model calls, no new facts, nothing to review")
    args = ap.parse_args()

    if args.verify:
        pool = json.loads(OUT.read_text())
        episodes = {e["episode_id"]: e
                    for e in json.loads(EPISODES.read_text())}
        keep = []
        for entry in pool:
            episode = episodes.get(entry.get("episode_id"))
            if episode and cites_its_moment(episode, entry.get("timestamp", ""),
                                            entry.get("text", "")):
                keep.append(entry)
            else:
                print(f"  dropped {entry.get('timestamp')} — "
                      f"{entry.get('text','')[:66]}")
        OUT.write_text(json.dumps(keep, ensure_ascii=False, indent=2))
        print(f"\n  {len(keep)}/{len(pool)} highlights cite a moment that "
              f"actually mentions them")
        return 0

    if args.relink:
        pool = json.loads(OUT.read_text())
        pool, added = relink(pool, json.loads(EPISODES.read_text()))
        OUT.write_text(json.dumps(pool, ensure_ascii=False, indent=2))
        have = sum(1 for h in pool if h.get("url"))
        print(f"  linked {added} more · {have}/{len(pool)} entries have a URL")
        return 0

    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    episodes = json.loads(EPISODES.read_text())
    if args.only:
        wanted = set(args.only)
        episodes = [e for e in episodes if e["episode_id"] in wanted]
        if not episodes:
            print(f"  none of {sorted(wanted)} are in {EPISODES.name}")
            return 1

    pool: list[dict] = []
    for i, episode in enumerate(episodes, 1):
        try:
            found = await highlights_for(
                client, settings.summary_model, episode, args.per_episode,
                prompt=FUNNY_PROMPT if args.kind == "funny" else None,
                kind=args.kind)
        except Exception as exc:                              # noqa: BLE001
            print(f"  [{i}/{len(episodes)}] FAILED {episode['title'][:40]}: {exc}")
            continue
        pool.extend(found)
        print(f"  [{i}/{len(episodes)}] {len(found)} from "
              f"{episode['title'][:44]}")

    if (args.kind == "funny" or args.only) and OUT.exists():
        # Appended, never replacing. A run for jokes must not throw away the
        # facts already approved, and a run for one new broadcast must not
        # throw away the other thirty-two — which is exactly what happened
        # the first time an episode was added after these guards existed.
        existing = json.loads(OUT.read_text())
        fresh = {(h["episode_id"], h["timestamp"]) for h in existing}
        pool = existing + [h for h in pool
                           if (h["episode_id"], h["timestamp"]) not in fresh]
    OUT.write_text(json.dumps(pool, ensure_ascii=False, indent=2))
    print(f"\n  {len(pool)} highlights -> {OUT.relative_to(ROOT)}")
    print("  Read them before enabling: these get posted unprompted, so a "
          "weak one is worse than none.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
