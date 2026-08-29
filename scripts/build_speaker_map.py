"""Turn anonymous voice clusters into names, for every episode at once.

    .venv/bin/python scripts/build_speaker_map.py
    .venv/bin/python scripts/build_speaker_map.py --report

Diarization gives you SPEAKER_00 and SPEAKER_01, renumbered differently in
every episode. Names are the missing half, and the trick is that the two
hosts are in all 33 shows while every guest is in one or two: cluster the
voice fingerprints across the whole archive, and the two that recur
everywhere are Ansem and Banks. Identified once, by ear, from stitched
clips built only of segments each cluster owns — a continuous sample
would have contained whoever spoke either side of it and proved nothing.

Everything else is deliberately left unknown. A guest heard in one
episode cannot be identified from recurrence, and guessing from the title
is how "Austin Federa said" ended up on a quote by somebody from a
different company. An unnamed speaker is a worse answer; a wrongly named
one is a false claim about a real person.

Writes data/speaker_map.json: episode -> segment index -> name. Nothing
here touches the embeddings, and this file is applied to `text_ts` and the
term index only, so retrieval is unchanged by construction.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from label_speakers import cluster  # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
FINGERPRINTS = ROOT / "data" / "speakers"
OUT = ROOT / "data" / "speaker_map.json"

# Identified by ear, 2026-08-28, from stitched cluster-only clips.
# Ordered by how many other episodes each voice appears in.
HOSTS = ["FaZe Banks", "Ansem"]

# How close a cluster must sit to a host's reference voice. Measured on
# this archive: two different people in the same episode never exceed
# 0.534, and the recurring host voices match each other above 0.85. The
# gap between those is wide, so this sits in the middle of it rather than
# at either edge.
SAME_PERSON = 0.85

# A cluster below this share of an episode is a fragment — a cough, a
# clip of music, a crosstalk artifact. The top four clusters hold 97% of
# an episode between them.
WORTH_LABELLING = 0.03

# How many consecutive segments one voice must hold before it earns a
# name. Without this the labels flip mid-thought during fast back-and-
# forth: one continuous run — "you put me onto the pump trade / I'm up a
# lot on pump / thank you again for that" — came out split across both
# hosts, and the half that mattered was given to the wrong one.
#
# The fingerprint of a two-second segment in a crosstalk stretch is a
# coin toss. Over four in a row it is not. So a name survives only where
# somebody actually held the floor, and the scrappy bits stay unlabelled
# — which is the honest description of them anyway.
MIN_RUN = 4

# How much a single segment must look like a host, and how far ahead of
# the other host it must be, before it earns a name.
#
# Names used to be assigned per cluster: the cluster's average voice was
# matched to a host and every segment in it inherited that name. On the
# one line that mattered most — "I put close to seven figures in
# Hyperliquid" — the cluster average said Ansem while the segments
# themselves scored 0.69 Banks against 0.25 Ansem, a margin of 0.44. The
# average was wrong and the evidence underneath it was right.
#
# Measured on this archive: a host's own segments run 0.44-0.69 against
# their reference and near zero against the other. A guest scores low
# against both. So a floor of 0.35 with a 0.15 margin keeps the confident
# ones and abstains everywhere else.
MIN_SIMILARITY = 0.35
MIN_MARGIN = 0.15


def hold_the_floor(sequence: list[str | None],
                   min_run: int = MIN_RUN) -> list[str | None]:
    """Drop any label that does not persist across min_run segments.

    Conservative on purpose. An unlabelled line reads as "somebody on the
    show said this", which is worse than a name and far better than the
    wrong name — the reverse trade is how a quote about being up seven
    figures got attributed to the host who was thanking the other one
    for it.
    """
    # A gap is not a speaker change. Segments that failed the confidence
    # bar carry no opinion, and treating them as a boundary meant
    # "Banks, gap, Banks, gap, Banks" — one man plainly holding the floor
    # — never reached a run of four and lost its name entirely. Only a
    # DIFFERENT name ends a run.
    out: list[str | None] = list(sequence)
    runs: list[tuple[str, list[int]]] = []
    for i, name in enumerate(sequence):
        if name is None:
            continue
        if runs and runs[-1][0] == name:
            runs[-1][1].append(i)
        else:
            runs.append((name, [i]))
    for _name, positions in runs:
        if len(positions) < min_run:
            for i in positions:
                out[i] = None
    return out


def episode_clusters():
    """(episode_id, share, centroid, segment indices) per dominant voice."""
    out = []
    for path in sorted(glob.glob(str(FINGERPRINTS / "*.npz"))):
        episode_id = Path(path).stem
        data = np.load(path)
        vectors = data["vectors"]
        index = (data["segment_index"] if "segment_index" in data
                 else np.arange(len(vectors)))
        if len(vectors) < 20:
            continue
        labels = cluster(vectors, 0.80)
        sizes = np.bincount(labels)
        for c in np.argsort(sizes)[::-1]:
            share = sizes[c] / len(labels)
            if share < WORTH_LABELLING:
                break
            where = np.where(labels == c)[0]
            centroid = vectors[where].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            out.append((episode_id, share, centroid, index[where]))
    return out


def host_references(clusters):
    """The two voices present in the most other episodes."""
    centroids = np.stack([c[2] for c in clusters])
    similarity = centroids @ centroids.T
    np.fill_diagonal(similarity, -1)
    spread = [
        len({clusters[j][0] for j in np.where(similarity[i] > SAME_PERSON)[0]
             if clusters[j][0] != clusters[i][0]})
        for i in range(len(clusters))
    ]
    picked: list[int] = []
    for i in np.argsort(spread)[::-1]:
        # Not the same person twice: a host's own clusters from different
        # episodes are near-identical and would otherwise fill both slots.
        if any(similarity[i, p] > SAME_PERSON for p in picked):
            continue
        picked.append(int(i))
        if len(picked) == len(HOSTS):
            break
    return [(HOSTS[n], clusters[i][2], spread[i])
            for n, i in enumerate(picked)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="print coverage and write nothing")
    args = ap.parse_args()

    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}
    clusters = episode_clusters()
    if not clusters:
        print("  no fingerprints — run label_speakers.py first")
        return 1

    references = host_references(clusters)
    print(f"\n  reference voices, from {len(clusters)} clusters across "
          f"{len({c[0] for c in clusters})} episodes:")
    for name, _, spread in references:
        print(f"     {name:12} recurs in {spread} other episodes")

    # Per segment, not per cluster. See MIN_SIMILARITY.
    names = [r[0] for r in references]
    matrix = np.stack([r[1] for r in references])
    mapping: dict[str, dict[str, str]] = {}
    labelled = total = 0
    for path in sorted(glob.glob(str(FINGERPRINTS / "*.npz"))):
        episode_id = Path(path).stem
        data = np.load(path)
        vectors, kept = data["vectors"], data["kept"]
        if len(vectors) < 20:
            continue
        total += len(vectors)
        rows: dict[str, str] = {}
        for j, segment_index in enumerate(kept):
            vector = vectors[j]
            vector = vector / (np.linalg.norm(vector) + 1e-9)
            scores = matrix @ vector
            order = np.argsort(scores)[::-1]
            best, runner = float(scores[order[0]]), float(scores[order[1]])
            if best < MIN_SIMILARITY or best - runner < MIN_MARGIN:
                continue                  # not confident enough to name
            rows[str(int(segment_index))] = names[order[0]]
        if rows:
            mapping[episode_id] = rows
            labelled += len(rows)

    # Smooth per episode, in segment order. Until now each cluster was
    # written independently, so nothing knew that two labels sat next to
    # each other — which is exactly the information needed to tell a
    # speaker holding the floor from a coin toss during crosstalk.
    before = labelled
    for episode_id, rows in list(mapping.items()):
        if not rows:
            continue
        order = sorted(int(i) for i in rows)
        span = range(min(order), max(order) + 1)
        sequence = [rows.get(str(i)) for i in span]
        kept = hold_the_floor(sequence)
        mapping[episode_id] = {str(i): name
                               for i, name in zip(span, kept, strict=True)
                               if name}
    labelled = sum(len(r) for r in mapping.values())
    mapping = {e: r for e, r in mapping.items() if r}
    print(f"\n  {before - labelled} labels dropped for not holding the "
          f"floor for {MIN_RUN} segments")

    print(f"  {labelled}/{total} dominant-voice segments labelled "
          f"({labelled / max(total, 1):.0%})")
    covered = len(mapping)
    print(f"  {covered}/{len(episodes)} episodes have at least one host "
          f"identified")
    missing = [e for e in episodes if e not in mapping]
    for episode_id in missing:
        print(f"     no host found  {episodes[episode_id]['title'][:52]}")

    if args.report:
        print("\n  --report: nothing written\n")
        return 0

    OUT.write_text(json.dumps(mapping, indent=0))
    print(f"\n  wrote {OUT.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
