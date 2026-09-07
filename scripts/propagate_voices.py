"""Carry a name a human already gave to every other show that voice is in.

    .venv/bin/python scripts/propagate_voices.py
    .venv/bin/python scripts/propagate_voices.py --apply

name_voices.py asks a person who a voice belongs to. That answer is worth
more than one episode: ECAPA embeddings are comparable across recordings,
so once somebody has been named anywhere, the same voice can be recognised
everywhere else without asking again. Mizkif was named in episode 2 and
went unlabelled in the episode titled after him.

The whole risk here is a confident wrong name, which is the failure this
archive does not survive -- "Austin Federa said" over a quote by someone
from another company is what made the pipeline refuse to guess in the
first place. So the bar is set from measurement, not from taste:

  * DIFFERENT people in this archive sit as high as 0.48 cosine
    (Al Dunlap vs Mizkif 0.483, Easy Eats vs Will Clemente 0.476). Any
    threshold near that is guessing. MIN_SCORE is 0.85.
  * A high score against a blurred prototype is not enough on its own, so
    the winner must also beat the runner-up by MIN_MARGIN. A voice that
    half-matches two people is not a match.
  * Small clusters are mostly crosstalk and backchannel. MIN_SEGMENTS
    drops them.
  * A person already labelled in an episode is skipped there: a human
    said where they speak, and this should not spread that name into
    segments the human did not give it to.

Dry run by default. --apply writes data/speaker_map.json, and prints what
it wrote, because a silent write to the file every citation depends on is
not something anyone should have to go looking for.

What this deliberately does NOT do is name a voice nobody has ever named.
There is no unsupervised path to a person's name in here; every name it
writes traces back to a human who listened.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPEAKERS = ROOT / "data" / "speakers"
MAP = ROOT / "data" / "speaker_map.json"

# Measured against this archive, not chosen. See the module docstring.
MIN_SCORE = 0.85
MIN_MARGIN = 0.35
MIN_SEGMENTS = 25
# How much of a cluster a name must already cover before that cluster is
# treated as evidence for the voiceprint. Below this the cluster is mixed
# and would blur the prototype it feeds.
OWNED = 0.6


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def load():
    """(named clusters, unnamed clusters) across every fingerprinted show."""
    smap = json.loads(MAP.read_text())
    named, unnamed = [], []
    for path in sorted(SPEAKERS.glob("*.json")):
        data = json.loads(path.read_text())
        episode, segments = data["episode_id"], data["segments"]
        labels = smap.get(episode, {})
        owners, sizes = defaultdict(Counter), Counter()
        for index, cluster in segments.items():
            sizes[str(cluster)] += 1
            if labels.get(index):
                owners[str(cluster)][labels[index]] += 1
        for cluster, centroid in data["centroids"].items():
            vector, size = unit(centroid), sizes[cluster]
            who = owners.get(cluster)
            if who and who.most_common(1)[0][1] / max(1, size) > OWNED:
                named.append((episode, cluster, who.most_common(1)[0][0], vector))
            elif not who and size >= MIN_SEGMENTS:
                unnamed.append((episode, cluster, size, vector))
    return smap, named, unnamed


def voiceprints(named) -> dict[str, np.ndarray]:
    """One vector per person, averaged over every cluster they own."""
    parts = defaultdict(list)
    for _, _, name, vector in named:
        parts[name].append(vector)
    return {name: unit(np.mean(vs, axis=0)) for name, vs in parts.items()}


def confirm(smap: dict, specs: list[str], write: bool) -> int:
    """Write names a person gave after listening, over the whole cluster.

    Unlike the automatic path this DOES overwrite existing labels, because
    a cluster the machine split off and a human then identified is more
    trustworthy than whatever the host-labelling pass left inside it. In
    the Mizkif episode three lines in his cluster carried host names, one
    of them his own introduction attributed to FaZe Banks. Refusing to
    overwrite would preserve exactly the errors this is here to fix.

    Every overwrite is printed. A label silently replaced in the file every
    citation depends on is worse than one that stays wrong in the open.
    """
    changed = replaced = 0
    for spec in specs:
        try:
            where, name = spec.split("=", 1)
            episode, cluster = where.rsplit(":", 1)
        except ValueError:
            print(f"  cannot read {spec!r}; want EPISODE:CLUSTER=NAME")
            return 1
        path = SPEAKERS / f"{episode}.json"
        if not path.exists():
            print(f"  no fingerprints for {episode}")
            return 1
        segments = json.loads(path.read_text())["segments"]
        bucket = smap.setdefault(episode, {})
        mine = [i for i, c in segments.items() if str(c) == cluster]
        if not mine:
            print(f"  no cluster {cluster} in {episode}")
            return 1
        was = Counter(bucket.get(i) for i in mine if bucket.get(i))
        for index in mine:
            if bucket.get(index) != name:
                if bucket.get(index):
                    replaced += 1
                bucket[index] = name
                changed += 1
        print(f"  {episode} cluster {cluster} -> {name.strip()}  "
              f"({len(mine)} segments)")
        for old, n in was.most_common():
            print(f"      overwrote {n} previously labelled {old}")
    if not write:
        print("\n  dry run. re-run with --apply to write these.")
        return 0
    MAP.write_text(json.dumps(smap, indent=1, sort_keys=True))
    print(f"\n  wrote {changed} labels ({replaced} corrections) to "
          f"{MAP.relative_to(ROOT)}")
    print("  re-embed to put the new names into retrieval.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the matches into data/speaker_map.json")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    # A person listened to the samples this script printed and said who it
    # is. That answer is bound to the clustering in data/speakers/*.json,
    # which is the clustering the samples were drawn from -- name_voices.py
    # re-clusters the .npz at its own threshold and numbers the voices
    # differently, so routing a confirmation through it could write the
    # name onto a different set of lines than the one that was heard.
    ap.add_argument("--confirm", action="append", default=[],
                    metavar="EPISODE:CLUSTER=NAME",
                    help="a human heard this cluster and says who it is")
    args = ap.parse_args()

    smap, named, unnamed = load()

    if args.confirm:
        return confirm(smap, args.confirm, write=args.apply)
    prints = voiceprints(named)
    if not prints:
        print("  nothing has been named yet — run name_voices.py first")
        return 1

    print(f"  voiceprints from {len(named)} named clusters: "
          + ", ".join(f"{n} ({len(list(1 for x in named if x[2] == n))})"
                      for n in sorted(prints)))
    print(f"  {len(unnamed)} unnamed clusters of {MIN_SEGMENTS}+ segments\n")

    matches = []
    for episode, cluster, size, vector in unnamed:
        ranked = sorted(((float(vector @ p), name)
                         for name, p in prints.items()), reverse=True)
        (score, name), (runner_up, _) = ranked[0], ranked[1]
        if score < args.min_score or score - runner_up < args.min_margin:
            continue
        # A human already placed this person in this episode. Their answer
        # stands; this does not widen it.
        if name in set(smap.get(episode, {}).values()):
            print(f"  skip  {episode[:14]:14} cl{cluster:>3}  {name} "
                  f"is already named here by hand")
            continue
        matches.append((score, score - runner_up, episode, cluster, size, name))

    if not matches:
        print("  no cluster clears the bar. nothing to write.")
        return 0

    print(f"  {len(matches)} match(es):\n")
    for score, margin, episode, cluster, size, name in sorted(matches, reverse=True):
        print(f"   {score:.3f}  margin +{margin:.2f}  {size:5} segs  "
              f"{episode[:14]:14} cl{cluster:>3}  ->  {name}")

    if not args.apply:
        print("\n  dry run. re-run with --apply to write these.")
        return 0

    written = 0
    for _, _, episode, cluster, _, name in matches:
        data = json.loads((SPEAKERS / f"{episode}.json").read_text())
        bucket = smap.setdefault(episode, {})
        for index, owner in data["segments"].items():
            if str(owner) == cluster and index not in bucket:
                bucket[index] = name
                written += 1
    MAP.write_text(json.dumps(smap, indent=1, sort_keys=True))
    print(f"\n  wrote {written} segment labels to {MAP.relative_to(ROOT)}")
    print("  re-embed to put the new names into retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
