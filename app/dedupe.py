"""Keep the same broadcast from being indexed twice.

The show goes out live and then reaches the index by two different roads:
the YouTube upload, and the X post of the same broadcast. Those roads do
not carry the same thing — the YouTube cut of an episode routinely drops
the guest interviews, and some segments only ever appear on X — so both
are worth having. But where they overlap, indexing both is worse than
indexing either.

Two copies of one passage is not merely wasteful. Retrieval returns a fixed
number of windows, so a duplicate spends one of those slots repeating a
moment instead of surfacing a different one, and an answer built from six
windows can end up reading two copies of the same sentence. It also makes
the *citation* a coin flip, which is the part that actually hurts: X has no
timestamp parameter for video, so half the time the answer would cite the
copy a reader cannot jump to while an identical passage sits on YouTube
with a working link.

Hence the precedence: where two episodes cover the same ground, the one
whose citations can jump wins. That is YouTube today, and the rule is
written in terms of the capability rather than the brand, so a platform
that gains deep links later slots in without a special case.

Direction matters and was the bug this file fixes. Guarding only at X
fetch time catches "X arrives when YouTube is already indexed" and misses
the reverse — an X broadcast indexed on Monday, the YouTube upload landing
Friday by cron, nothing checking. Running this after ANY fetch covers both.
"""

from __future__ import annotations

import random
import re

# Platforms whose deep links carry a timestamp. A citation into one of
# these can be clicked; a citation anywhere else can only be read.
_SEEKABLE = {"youtube", "spotify"}

# Share of sampled phrases that must appear in the other transcript before
# two episodes are treated as the SAME RECORDING rather than two cuts that
# share material.
#
# 0.90, and the number was learned the hard way. At 0.35 this proposed
# deleting eight episodes from a catalogue that answers correctly today,
# including the one behind a working example question on the home page.
#
# The reason is that partial overlap is normal here, not exceptional: the
# channel publishes a long episode and several short titled cuts from the
# same session. Measured across the working index, ordinary pairs reach 66%,
# 87%, even 94%. And the overlap is never the interesting part — the
# "BlackRock Told Tristan Thompson" clip is 65% contained in another
# episode, and the word BlackRock appears in it three times and in the other
# one zero times. Dropping the clip would have deleted exactly the third
# that made it worth having.
#
# So the bar is "this is the same recording twice", not "these share
# material". Below it, both copies stay.
SAME_RECORDING = 0.90

_CLEAN = re.compile(r"[^a-z0-9 ]")
_SPACE = re.compile(r"\s+")


def _flatten(segments: list[dict]) -> str:
    text = " ".join(s.get("text", "") for s in segments)
    return _SPACE.sub(" ", _CLEAN.sub(" ", text.lower())).strip()


def overlap(a: list[dict], b: list[dict], samples: int = 300) -> float:
    """Roughly what share of `a` also appears in `b`.

    Sampled rather than exhaustive — this only has to separate "the same
    recording" from "a different one", and an exact diff of two five-hour
    transcripts costs far more than that decision is worth.

    Asymmetric on purpose. A 60-minute interview fully contained in a
    five-hour upload scores ~1.0 in this direction and ~0.2 in the other,
    and the first number is the one that answers "is this already indexed".
    """
    mine = _flatten(a).split()
    theirs = _flatten(b)
    if len(mine) < 100 or not theirs:
        return 0.0
    rng = random.Random(7)          # deterministic: reruns must agree
    picks = rng.sample(range(0, len(mine) - 8), min(samples, len(mine) - 8))
    hits = sum(1 for i in picks if " ".join(mine[i:i + 7]) in theirs)
    return hits / len(picks)


def _rank(episode: dict) -> tuple[int, int]:
    """Sort key deciding which copy survives. Higher wins.

    Seekable first, because a citation you can click is the whole feature.
    Longer second, so between two equally linkable copies the fuller one is
    kept rather than a clip of it.
    """
    seekable = 1 if episode.get("platform") in _SEEKABLE else 0
    return (seekable, len(episode.get("segments") or []))


def dedupe(episodes: list[dict],
           threshold: float = SAME_RECORDING) -> tuple[list[dict], list[dict]]:
    """(kept, dropped) — only near-identical recordings are dropped.

    Compares every pair once. At a few dozen episodes that is trivial; if
    this ever holds thousands, bucket by date before pairing.
    """
    # Strongest first, so a survivor is always compared against something
    # already established rather than the other way round.
    ordered = sorted(episodes, key=_rank, reverse=True)
    kept: list[dict] = []
    dropped: list[dict] = []

    for episode in ordered:
        duplicate_of = None
        for winner in kept:
            if overlap(episode.get("segments") or [],
                       winner.get("segments") or []) >= threshold:
                duplicate_of = winner
                break
        if duplicate_of is None:
            kept.append(episode)
        else:
            episode = dict(episode)
            episode["_duplicate_of"] = duplicate_of["episode_id"]
            dropped.append(episode)

    # Restore a stable, human-meaningful order for the file on disk.
    kept.sort(key=lambda e: (e.get("published_at") or ""), reverse=True)
    return kept, dropped


def describe(dropped: list[dict]) -> str:
    if not dropped:
        return "no duplicate coverage found"
    lines = [f"dropped {len(dropped)} duplicate episode(s):"]
    for episode in dropped:
        lines.append(f"    {episode['episode_id']:24s} "
                     f"{episode.get('title', '')[:44]:46s} "
                     f"-> already covered by {episode['_duplicate_of']}")
    return "\n".join(lines)
