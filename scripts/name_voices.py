"""Play a voice, ask a human who it is, write down the answer.

    .venv/bin/python scripts/name_voices.py --episode x-2095611959161082020
    .venv/bin/python scripts/name_voices.py --episode <id> --name 3 "Mizkif"

The archive labels the two hosts and nobody else, on purpose. They are in
all 33 shows, so the voice that recurs everywhere is identifiable by
recurrence alone; a guest appears once and cannot be. Guessing the guest
from the episode title is what put "Austin Federa said" on a quote by
somebody from a different company, so the pipeline refuses to guess and
leaves 63% of speech unattributed.

Recurrence is not the only kind of evidence, though. A person who knows
the show can name a voice in about four seconds. This does the part a
machine is good at -- find the distinct voices, work out which are already
known, cut a sample of each -- and asks a human for the part a human is
good at.

The sample is stitched from segments the cluster OWNS, never a continuous
stretch. A continuous stretch contains whoever spoke either side of it, so
a listener would be naming a conversation rather than a voice. That is the
same reason the hosts were identified from stitched clips in the first
place.

Nothing is written until a name is given, and a name is only ever written
for the cluster it was given for. There is no inference step here at all,
which is the point: this is the one path to a guest's name that does not
involve a guess.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.clipper import stamp                          # noqa: E402
from label_speakers import cluster                      # noqa: E402

EPISODES = ROOT / "data" / "episodes.json"
FINGERPRINTS = ROOT / "data" / "speakers"
MAP = ROOT / "data" / "speaker_map.json"
AUDIO = ROOT / "audio"
SAMPLES = Path("/tmp/voice_samples")

# How much of an episode a voice must hold before it is worth a human's
# time. Below this it is usually a caller, a clip played on air, or a
# crosstalk artifact -- and naming those wrongly costs the same as naming
# a guest wrongly.
MIN_SHARE = 0.02
SLICES = 6          # samples stitched into one clip
SLICE_SECONDS = 3.5


def _seconds(stamp: str) -> float:
    """h:mm:ss, m:ss or plain seconds — whatever the watcher typed."""
    parts = [float(p) for p in str(stamp).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def load_episode(episode_id: str) -> dict:
    for e in json.loads(EPISODES.read_text()):
        if e["episode_id"] == episode_id:
            return e
    raise SystemExit(f"  no episode {episode_id!r}")


def deep_link(episode: dict, seconds: float) -> str:
    """Where to go and listen, if the clip alone is not enough."""
    sec = int(seconds)
    url = episode["url"]
    if episode.get("platform") == "youtube":
        return f"{url}{'&' if '?' in url else '?'}t={sec}s"
    return f"{url}{'&' if '?' in url else '?'}t={sec}"


def stitch(audio: Path, starts: list[float], out: Path) -> bool:
    """One clip of several disjoint slices, so only this voice is in it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, t in enumerate(starts):
        piece = out.parent / f"{out.stem}-{i}.wav"
        done = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "quiet", "-y",
             "-ss", f"{t:.2f}", "-t", str(SLICE_SECONDS), "-i", str(audio),
             "-ac", "1", "-ar", "16000", str(piece)],
            capture_output=True)
        if done.returncode == 0 and piece.exists():
            parts.append(piece)
    if not parts:
        return False
    listing = out.parent / f"{out.stem}.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "quiet", "-y", "-f", "concat",
         "-safe", "0", "-i", str(listing), "-c", "copy", str(out)],
        capture_output=True)
    for p in parts:
        p.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)
    return out.exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--name", nargs=2, metavar=("CLUSTER", "NAME"),
                    help="record that cluster N is this person")
    ap.add_argument("--min-share", type=float, default=MIN_SHARE)
    # The fingerprinting threshold is tuned for purity -- it would rather
    # split one person into forty clusters than put two people in one. That
    # is right when a machine is deciding and useless when a human is: on
    # one episode it produced 893 voices, the largest holding 10% of the
    # show, so naming one would label a tenth of a tenth of the speech.
    #
    # Grouped loosely enough to be worth a person's time instead. Measured
    # on ep 18 against the host labels, no cluster mixed the two hosts at
    # any value up to 0.90. That check has a hole in it -- the only labels
    # to measure against are hosts, so a guest absorbed into a host's
    # cluster would still score perfectly -- which is what the listening
    # is for. A sample stitched from six places in a cluster makes two
    # people in one cluster obvious in about a second.
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="grouping distance; lower splits, higher merges")
    ap.add_argument("--window", nargs=2, metavar=("FROM", "TO"),
                    help="only name lines in this range, e.g. 2:05:00 2:31:05")
    args = ap.parse_args()

    episode = load_episode(args.episode)
    npz = FINGERPRINTS / f"{args.episode}.npz"
    if not npz.exists():
        raise SystemExit(f"  no fingerprints for {args.episode}")
    data = np.load(npz)
    labels = cluster(data["vectors"], args.threshold)
    kept = list(data["kept"])
    segments = episode["segments"]
    by_cluster: dict[int, list[int]] = collections.defaultdict(list)
    for idx, c in zip(kept, labels):
        by_cluster[int(c)].append(int(idx))

    known = (json.loads(MAP.read_text()) if MAP.exists() else {}
             ).get(args.episode, {})

    if args.name:
        which, person = int(args.name[0]), args.name[1].strip()
        if which not in by_cluster:
            raise SystemExit(f"  no cluster {which} in this episode")
        idxs = by_cluster[which]

        # A cluster is "one voice", not "one person all night". Voice 8 of
        # ep 18 held 208 segments inside Will Clemente's slot and 20
        # scattered elsewhere, one of them at 20:46 -- an hour and a half
        # before he joined. Writing his name across the whole cluster
        # would trade a wrong host for a wrong guest.
        #
        # So a window can be given, and then a name is written only where
        # the voice and the presence agree: the clustering says these
        # lines are one speaker, the person who watched says who was in
        # the room. Outside the window the lines are left as they were.
        skipped = 0
        if args.window:
            a, b = (_seconds(x) for x in args.window)
            inside = [i for i in idxs
                      if i < len(segments) and a <= segments[i]["t"] <= b]
            skipped = len(idxs) - len(inside)
            idxs = inside
        if not idxs:
            raise SystemExit("  nothing in that window — nothing written")

        full = json.loads(MAP.read_text()) if MAP.exists() else {}
        entry = full.setdefault(args.episode, {})
        overwritten = collections.Counter(entry.get(str(i)) for i in idxs
                                          if entry.get(str(i)))
        for idx in idxs:
            entry[str(idx)] = person
        MAP.write_text(json.dumps(full))
        print(f"  cluster {which} -> {person!r} on {len(idxs)} lines "
              f"of {episode['title'][:40]}")
        if skipped:
            print(f"  {skipped} lines left alone, outside the window")
        for was, n in overwritten.most_common():
            print(f"  replaced {n} lines that said {was!r}")
        return 0

    audio = AUDIO / f"{args.episode}.mp3"
    if not audio.exists():
        raise SystemExit(f"  no audio cached at {audio}")
    SAMPLES.mkdir(exist_ok=True)
    print(f"\n  {episode['title'][:60]}")
    print(f"  {len(by_cluster)} voices, {len(kept)} fingerprinted lines\n")

    for which, idxs in sorted(by_cluster.items(),
                              key=lambda kv: -len(kv[1])):
        share = len(idxs) / max(1, len(kept))
        if share < args.min_share:
            continue
        # A cluster the map already names is shown, not offered: it is
        # how the listener calibrates that the clips sound like who they
        # are supposed to sound like before trusting the unnamed ones.
        # How many lines actually carry that name, not just which name
        # wins among the few that do. Printing a bare "already named:
        # FaZe Banks" for a cluster where 2 lines of 228 say Banks reads
        # as a settled fact and is nearly the opposite -- it sent me
        # looking for a mislabelling that was two lines wide.
        names = collections.Counter(known.get(str(i)) for i in idxs
                                    if known.get(str(i)))
        settled = None
        if names:
            who, n = names.most_common(1)[0]
            settled = f"{who} on {n} of {len(idxs)} lines"

        spread = idxs[::max(1, len(idxs) // SLICES)][:SLICES]
        starts = [segments[i]["t"] for i in spread if i < len(segments)]
        out = SAMPLES / f"{args.episode}-voice{which}.wav"
        ok = stitch(audio, starts, out)

        head = f"  voice {which}   {share*100:5.1f}% of the episode"
        print(head + (f"   — already named: {settled}" if settled else ""))
        if ok:
            print(f"     listen   afplay {out}")
        # Every slice, not just the first. The clip is stitched from six
        # places in the episode, so a listener who doubts one can check
        # any of the others -- and if two of them are different people,
        # the cluster is wrong and these are how it gets found.
        for n, t in enumerate(starts, 1):
            said = next((segments[i]["text"].strip()[:58] for i in spread
                         if i < len(segments)
                         and abs(segments[i]["t"] - t) < 0.01
                         and segments[i]["text"].strip()), "")
            print(f"     {n}. {stamp(t):>8}  {deep_link(episode, t)}")
            if said:
                print(f"            \"{said}\"")
        print()

    print("  name one with:\n"
          f"    scripts/name_voices.py --episode {args.episode} "
          "--name <voice> \"Their Name\"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
