"""Turn "who was on air" into "who said this line".

    .venv/bin/python scripts/write_guest_labels.py
    .venv/bin/python scripts/write_guest_labels.py --only x-2095611959161082020

read_guest_windows.py reads the show's own lower third and gets a name, a
role and the interval somebody was on air. That is not the same as
knowing who spoke: every tile is captioned at once, and the host is
usually talking at the moment a guest's banner goes up -- measured, Banks
is speaking for the ninety seconds before Will Clemente's first word.
Labelling every line in a window with the guest's name would put the
guest's name on the host's sentences.

What makes it safe is that the voices are already separated. label_speakers.py
gives every segment a cluster id from its ECAPA fingerprint, and
build_speaker_map.py names the two clusters that recur across the whole
archive -- the hosts. So inside a guest's window, a large cluster that is
NOT a host cluster is the guest, and the banner says who that is. Two
independent kinds of evidence: the audio separates the speakers, the
screen names them.

Deliberately conservative. The clustering is fragmented -- 893 distinct
clusters over 2,767 segments on ep 18 -- so only clusters with at least
MIN_SEGMENTS lines are claimed and the long tail is left unlabelled. An
unnamed line is a worse answer; a wrongly named one is a false claim
about a real person, and that is the failure this whole effort exists to
avoid.

Measured on ep 18's broadcast against the hand-built map: 278 segments
claimed, 278 agreeing, 0 disagreeing, 7 that the map had left blank.

Writes data/guest_labels.json, which is NOT merged into
data/speaker_map.json. The map is what a human vouched for by ear;
keeping them apart means either can be regenerated without losing the
distinction, and apply_speaker_labels.py lets the hand map win ties.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPISODES = ROOT / "data" / "episodes.json"
SPEAKER_MAP = ROOT / "data" / "speaker_map.json"
GUEST_WINDOWS = ROOT / "data" / "guest_windows.json"
SPEAKERS_DIR = ROOT / "data" / "speakers"
OUT = ROOT / "data" / "guest_labels.json"

# A cluster this big inside a guest's window is a person. Smaller ones
# are fingerprint fragments -- a couple of lines split off by crosstalk
# or a cough -- and claiming them buys a handful of labels at the price
# of not knowing whether they are right.
MIN_SEGMENTS = 8

# How many labelled lines make a cluster "a host's". Low on purpose: a
# host cluster wrongly treated as a guest's is a wrong name on a real
# person, while a guest cluster wrongly excluded is only a missing one.
HOST_EVIDENCE = 3

HOSTS = ("Ansem", "FaZe Banks")


def title_case(name: str) -> str:
    """The banner shouts. Stored names are ordinary case, to sit beside
    the hand map's "FaZe Banks" rather than next to it in caps."""
    fixed = []
    for word in name.split():
        if word.isupper() and len(word) > 1 and word.isalpha():
            fixed.append(word.capitalize())
        else:
            fixed.append(word)
    return " ".join(fixed)


def host_clusters(clusters: dict[str, int], hand: dict[str, str]) -> set[int]:
    """Cluster ids the hand map already attributes to a host."""
    counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for idx, cid in clusters.items():
        who = hand.get(idx)
        if who:
            counts[who][cid] += 1
    out: set[int] = set()
    for host in HOSTS:
        out |= {cid for cid, n in counts[host].items() if n >= HOST_EVIDENCE}
    return out


def labels_for(episode: dict, windows: list[dict], clusters: dict[str, int],
               hand: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Segment index -> guest name, for one episode."""
    # The same numbering apply_speaker_labels.py uses: non-empty text
    # only, in order. Anything else puts a name on the wrong line.
    numbered = [s for s in episode["segments"] if (s.get("text") or "").strip()]
    hosts = host_clusters(clusters, hand)
    out: dict[str, str] = {}
    notes: list[str] = []
    for window in windows:
        start, end = window["start"], window["end"]
        inside = collections.Counter()
        for i, segment in enumerate(numbered):
            if start <= segment["t"] <= end:
                cid = clusters.get(str(i))
                if cid is not None and cid not in hosts:
                    inside[cid] += 1
        claimed = {cid for cid, n in inside.items() if n >= MIN_SEGMENTS}
        if not claimed:
            notes.append(f"{window['name']}: no cluster over "
                         f"{MIN_SEGMENTS} lines — left unlabelled")
            continue
        name = title_case(window["name"])
        n = 0
        for i, segment in enumerate(numbered):
            if start <= segment["t"] <= end and clusters.get(str(i)) in claimed:
                # The hand map wins: a human said who this was.
                if str(i) not in hand:
                    out[str(i)] = name
                n += 1
        notes.append(f"{name}: clusters {sorted(claimed)} -> {n} lines")
    return out, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[],
                    metavar="EPISODE_ID")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    episodes = {e["episode_id"]: e for e in json.loads(EPISODES.read_text())}
    hand_all = json.loads(SPEAKER_MAP.read_text())
    windows_all = json.loads(GUEST_WINDOWS.read_text())

    todo = [e for e in windows_all if windows_all[e]]
    if args.only:
        todo = [e for e in todo if e in args.only]
    if not todo:
        print("no episodes with guest windows — run read_guest_windows.py")
        return 1

    result: dict[str, dict[str, str]] = {}
    if OUT.exists() and not args.dry_run:
        result = json.loads(OUT.read_text())

    total = 0
    for eid in todo:
        fingerprints = SPEAKERS_DIR / f"{eid}.json"
        if not fingerprints.exists():
            print(f"{eid}: no voice fingerprints — "
                  f"run label_speakers.py --only {eid}")
            continue
        clusters = json.loads(fingerprints.read_text())["segments"]
        hand = hand_all.get(eid, {})
        # Without a hand map there are no known host clusters, so
        # host_clusters() returns nothing, every cluster looks like a
        # guest's, and the banner's name would be pinned to the hosts'
        # own lines. Refusing is the only safe answer: an unnamed line
        # is a worse answer, a wrongly named one is a false claim about
        # a real person.
        if not hand:
            print(f"{eid}: no speaker map — skipped. The hosts' clusters "
                  f"are unknown, so nothing here can be attributed "
                  f"safely. Run build_speaker_map.py first.")
            continue
        labels, notes = labels_for(episodes[eid], windows_all[eid],
                                   clusters, hand)
        print(f"\n{eid}  {episodes[eid]['title'][:54]}")
        for note in notes:
            print(f"    {note}")
        print(f"    {len(labels)} new labels "
              f"({len(hand)} already named by hand)")
        if labels:
            result[eid] = labels
            total += len(labels)

    if args.dry_run:
        print(f"\n[dry run] {total} labels, nothing written")
        return 0
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT.relative_to(ROOT)}: "
          f"{total} labels across {len(result)} episode(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
