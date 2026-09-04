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
#
# X broadcasts are in here now. They always belonged: ?t=<seconds> opens
# the player at that second, verified against three broadcasts. While they
# were excluded, this ranking preferred the YouTube copy of every episode —
# and the YouTube copy is the CUT, roughly a third shorter than the
# broadcast it came from. So the rule that was meant to keep the clickable
# copy was quietly keeping the shorter one and deleting the full show.
#
# X posts carry platform "other" rather than "x", which is what the
# ingester writes; matching on the id prefix would be tidier but this is
# the field the rest of the pipeline reads.
_SEEKABLE = {"youtube", "spotify", "other"}

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

    Now that broadcasts count as seekable, the second key is what actually
    decides an X/YouTube pair — and it picks the broadcast, because the
    upload is a cut of it. That is the right way round: the cut is where
    the Squire interview went missing.
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


# ─── grouping the same show ───────────────────────────────────────────────
#
# Separate from dedupe() above, and deliberately so. That decides what is
# worth INDEXING, and its answer is "both copies": the YouTube cut drops
# the guest interviews and some segments only ever aired on X, so throwing
# either away loses real coverage.
#
# This decides what is worth LISTING, and the answer there is one card per
# show. The episodes page rendered 38 cards while the header said 18, with
# the 20 August broadcast appearing three times, because the page was
# showing files and the header was counting shows.
#
# Matched on three-word runs shared by the pair, and on nothing else. Two
# Whisper passes over the same audio produce similar text and never
# identical text, so the measure has to tolerate that:
#
#   exact 6-grams     too brittle. Two passes share almost no six-word run,
#                     so nothing merged at all.
#   word overlap      too loose. Two hours of crypto talk share a
#                     vocabulary, so unrelated episodes merged.
#   rare words        worked, and depended on the whole corpus to decide
#                     what "rare" meant. The same pair then grouped
#                     differently on two machines with different data
#                     files, and production listed 20 shows while this
#                     machine counted 18.
#
# Three-word runs need neither a corpus nor a vocabulary. Measured over
# known pairs: the same show scores 0.73 to 0.82, different shows 0.09 to
# 0.11, and the band between is empty. That is a property of the pair
# alone, so two machines cannot disagree.

import collections
import datetime

# Where to cut, in the middle of a gap seven times wider than either
# cluster. Nothing observed lands between 0.11 and 0.73.
_SAME_SHOW = 0.35
# A cut usually goes up the next day; one went up four days later.
_SAME_SHOW_DAYS = 6
_RUN = 3

_A_TOKEN = re.compile(r"[a-z']+")


def _runs(episode: dict) -> set[int]:
    """Hashes of every three-word run, which is cheaper than holding the
    runs themselves: a four-hour show is around forty thousand of them."""
    words = _A_TOKEN.findall(
        " ".join(s.get("text", "") for s in episode.get("segments") or []).lower())
    return {hash(tuple(words[i:i + _RUN]))
            for i in range(max(0, len(words) - _RUN))}


def _length(episode: dict) -> float:
    return max((s.get("t", 0) for s in episode.get("segments") or []), default=0)


def _aired(episode: dict) -> datetime.date | None:
    stamp = (episode.get("published_at") or episode.get("date") or "")[:10]
    try:
        return datetime.date.fromisoformat(stamp)
    except ValueError:
        return None


def same_show(a: dict, b: dict) -> bool:
    """Whether these two files are the same broadcast.

    Depends only on the two of them, which is the point.
    """
    when_a, when_b = _aired(a), _aired(b)
    if not when_a or not when_b:
        return False
    if abs((when_a - when_b).days) > _SAME_SHOW_DAYS:
        return False
    first, second = _runs(a), _runs(b)
    if not first or not second:
        return False
    smaller, larger = ((first, second) if len(first) <= len(second)
                       else (second, first))
    return len(smaller & larger) / len(smaller) > _SAME_SHOW


def group_by_show(episodes: list[dict]) -> list[list[dict]]:
    """Episodes clustered so that each cluster is one broadcast.

    Merged transitively, because one show can have four cuts.
    """
    marks = {e["episode_id"]: _runs(e) for e in episodes}
    parent = {e["episode_id"]: e["episode_id"] for e in episodes}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(episodes):
        for b in episodes[i + 1:]:
            first, second = marks[a["episode_id"]], marks[b["episode_id"]]
            if not first or not second:
                continue
            when_a, when_b = _aired(a), _aired(b)
            if not when_a or not when_b:
                continue
            if abs((when_a - when_b).days) > _SAME_SHOW_DAYS:
                continue
            smaller, larger = ((first, second) if len(first) <= len(second)
                               else (second, first))
            if len(smaller & larger) / len(smaller) > _SAME_SHOW:
                ra, rb = root(a["episode_id"]), root(b["episode_id"])
                if ra != rb:
                    parent[ra] = rb

    clusters: dict[str, list[dict]] = collections.defaultdict(list)
    for e in episodes:
        clusters[root(e["episode_id"])].append(e)
    return list(clusters.values())


def canonical_episode_ids(episodes: list[dict]) -> set[str]:
    """One id per show: the longest cut, which is the full broadcast."""
    return {max(c, key=_length)["episode_id"] for c in group_by_show(episodes)}


# The show numbers its own episodes, and both copies of a broadcast carry
# the number: "LIVE W/ WILL CLEMENTE ... Market Bubble Ep 18" and "We are
# entering a SUPERCYCLE | Market Bubble #18" are one evening.
#
# This exists because the transcript matcher cannot see an episode that is
# not in episodes.json, and a show ingested since the last deploy is in
# exactly that position: its summary is written to Pinecone by the ingest,
# the data file ships with the image, and until they meet the newest show
# is listed twice. That is what Ep 18 did on the night it aired.
#
# Parsed from 18 of the 20 titles in the archive, and no two different
# shows share a number, so it is safe to treat a shared number as proof of
# a shared evening. The two it cannot parse simply fall back to the
# transcript matcher, which is the stronger test when it can run at all.
_EPISODE_NUMBER = re.compile(
    r"(?ix)(?: market\s+bubble | ep(?:isode)? )\s*\#?\s*(\d{1,2})\b")


def episode_number(title: str) -> int | None:
    """The show's own number for an episode, from its title."""
    found = _EPISODE_NUMBER.search(title or "")
    return int(found.group(1)) if found else None
