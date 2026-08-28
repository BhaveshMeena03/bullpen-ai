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
import os
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

# Subjects this account does not post, whatever the model decided.
#
# These entries go out unprompted, under the show's name, with nobody
# reading them first — so the prompt cannot be the only thing standing
# between a transcript and a public post. Loosening one prompt rule (the
# hosts roasting each other is fair game, which it is) immediately
# surfaced a suicide joke, a stabbing bit, a slur, an antidepressant
# gag, and one host calling another a liar by name. Every one of them
# passed the rule it was tested against.
#
# So the prompt asks, and this decides. A funny moment wrongly dropped
# costs nothing; there are others. The reverse is not recoverable.
_UNPOSTABLE = re.compile(
    r"""(?ix)
      # self-harm, in any of the joking registers people use for it
      \b(?: kill\s+(?:my|him|her|them)self | suicide | suicidal
          | jump(?:ing)?\s+off | hang\s+(?:my|him|her)self
          | end\s+(?:it\s+all|my\s+life) | self[-\s]?harm | kms )\b
      # violence against a person
    | \b(?: stab | stabbing | stabbed | shoot(?:ing)? | shot\s+(?:him|her|them)
          | murder | kill(?:ing)?\s+(?:him|her|them|you|people)
          | rape | raped | assault(?:ed|ing)? | strangle | beat\s+(?:him|her|them)\s+up )\b
      # slurs and demeaning terms, including ones used self-deprecatingly
    | \b(?: retard(?:ed|s)? | tranny | fag(?:got)?s? | n[i1]gg[ae]r?s?
          | midget | spastic | cripple )\b
      # medication, mental health, addiction — rule 3a, made literal
    | \b(?: lexapro | lexa\s*pro | prozac | zoloft | xanax | adderall | ozempic
          | antidepressant | ssri | rehab | overdose | od(?:'?d|ed)
          | addict(?:ed|ion)? | withdrawal | relapse
          | depress(?:ed|ion) | bipolar | psychiatric )\b
      # illness
    | \b(?: cancer | tumou?r | chemo(?:therapy)? | terminal\s+illness
          | stroke | seizure )\b
      # crime committed against someone
    | \b(?: robbed | robbery | mugged | kidnap(?:ped|ping)? | burglar(?:y|ized)
          | held\s+at\s+gunpoint | swatted )\b
      # a named person accused of dishonesty: a joke to the room, a
      # defamation-shaped sentence once this account republishes it alone
    | \b(?: lied | lying | liar | fraud(?:ster)? | scammer | scammed
          | rug(?:ged|ging|ger)? | stole | stealing | thief )\b
      # how somebody looks. The original prompt banned this outright and
      # the rewrite kept it only for people outside the conversation —
      # which let through a co-host told he was "looking pretty busted
      # today". Being in the room makes it fair between them; it does not
      # make it this account's to repeat.
    | \b(?: busted | haggard | washed | looking\s+rough | ugly
          | fat | overweight | balding )\b
      # money somebody lost. Also in the original and also dropped. A loss
      # is not funnier for being large, and the reply calls it a joke.
    | \b(?: swindled | lost\s+(?:a\s+)?(?:huge|most|his|her|their|\$|
                                        fortune|fund|money|everything)
          | wrote\s+(?:it\s+)?off | blew\s+(?:his|her|their)
          | wiped\s+out | went\s+to\s+zero )
      # markup the model leaves in when it censors a quote itself: this
      # would post the word "[expletive]" to the timeline.
    | \[ (?:expletive|redacted|inaudible|laughs?) \]
    """)


def safe_to_post(text: str) -> bool:
    """Would this be all right going out on its own, with no one checking?

    Applies to facts as much as jokes: both get posted unprompted, and a
    fact about somebody's illness is not improved by being true.
    """
    return not _UNPOSTABLE.search(text or "")


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

3a. Prefer moments that are funny on their own terms — a story, a claim, \
an admission, a bit that goes somewhere stupid. Something dark said for \
shock will be filtered out later anyway, so it is a wasted pick.

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

    # Thinking off, and a budget with room to spare.
    #
    # Thinking is billed against max_tokens, so a long transcript and a
    # prompt full of rules can spend the entire budget reasoning and return
    # stop_reason=max_tokens with a single thinking block and no text. From
    # the outside that is indistinguishable from "no moment in this episode
    # qualifies", and three consecutive prompt rewrites were spent chasing
    # a refusal that was actually a truncation. Raising 1024 to 4000 only
    # moved the ceiling; the next run hit that too.
    #
    # This is extraction, not reasoning: pick lines out of a transcript and
    # write them in a fixed format. With thinking disabled the same call
    # answers in about 200 tokens instead of exhausting 4000.
    response = await client.messages.create(
        model=model, max_tokens=4000,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content":
                   f"{(prompt or PROMPT).format(n=n)}\n\n"
                   f"<transcript>\n{sampled}\n</transcript>"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    # Say so out loud. A silent empty answer reads as a clean "nothing
    # here" and is the one failure this script cannot afford to look calm.
    if not text.strip():
        print(f"     !! no text returned (stop_reason={response.stop_reason},"
              f" {response.usage.output_tokens} output tokens) — this is a"
              f" truncated call, not an episode without highlights")
    if os.environ.get("HIGHLIGHT_DEBUG"):
        print(f"  [stop_reason={response.stop_reason} "
              f"blocks={[b.type for b in response.content]} "
              f"out_tokens={response.usage.output_tokens}]")
        for ln in text.splitlines():
            print(f"    | {ln}")

    out = []
    for line in text.splitlines():
        if "|" not in line:
            if line.strip() and os.environ.get("HIGHLIGHT_DEBUG"):
                print(f"     [no pipe] {line[:70]}")
            continue
        said, _, stamp = line.rpartition("|")
        said, stamp = said.strip(" -–—"), stamp.strip().strip("[]")
        # The model sometimes wraps its answer in the tag from the format
        # block. Four of these reached the live pool and would have posted
        # "<sentence>…</sentence>" to the timeline.
        said = re.sub(r"</?[a-z_]+>", "", said).strip()
        if len(said) < 40 or not stamp:
            if os.environ.get("HIGHLIGHT_DEBUG"):
                print(f"     [too short / no stamp] {said[:60]!r} {stamp!r}")
            continue
        # Rule 1 of the prompt is to name the person. Only genuinely
        # anonymous attributions are rejected — a one-letter name is a
        # nickname on this show, not a missing source: Banks calls Ansem Z.
        # Every way the model opens a sentence when it does not know who
        # spoke. The first four were the original list; the rest each
        # reached a pool — "An anonymous guest claims to have bought
        # Bitcoin around $1.50" was one entry away from being posted as a
        # fact by an account whose entire promise is that it can tell you
        # who said a thing and when.
        if re.match(r"""(?ix) ^ (?: someone | a\s+guest | one\s+host
                                 | an?\s+host | the\s+host | the\s+guest
                                 | the\s+speaker | an?\s+speaker
                                 | an\s+anonymous | an?\s+unnamed
                                 | the\s+(?:finance|financial)\s+advisor
                                 | a\s+co-?host | bro )\b""", said):
            if os.environ.get("HIGHLIGHT_DEBUG"):
                print(f"     [unnamed] {said[:60]}")
            continue
        # And the model hedging about who spoke, anywhere in the sentence.
        # One of these reached the pool complete with its own parenthetical:
        # "(unnamed speaker's estimate, though speaker identity unclear from
        # transcript)" — which is the admission that it broke rule 1, left
        # inside the thing that would have been posted.
        if _UNSURE.search(said):
            if os.environ.get("HIGHLIGHT_DEBUG"):
                print(f"     [hedged attribution] {said[:60]}")
            continue
        # The subject filter runs last of the cheap checks and first in
        # authority: whatever the prompt allowed, this decides.
        if not safe_to_post(said):
            print(f"     dropped (not ours to post): {said[:60]}")
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
    ap.add_argument("--seekable-only", action="store_true",
                    help="only use episodes whose link can open at the "
                         "moment itself. X ignores every timestamp "
                         "parameter — verified in a browser, currentTime "
                         "stays 0 on a four and a half hour video — so a "
                         "citation there can only say 'scrub to 1:47:12'. "
                         "Fine for a fact somebody asked about. Useless for "
                         "a joke, where the punchline is the whole payload "
                         "and nobody scrubs four hours to reach it.")
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
    if args.seekable_only:
        before = len(episodes)
        episodes = [e for e in episodes if e.get("platform") == "youtube"]
        print(f"  {len(episodes)} of {before} episodes can link to the exact "
              f"second; the rest are X broadcasts and are skipped")
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
