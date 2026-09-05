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
    args = ap.parse_args()

    episode = load_episode(args.episode)
    npz = FINGERPRINTS / f"{args.episode}.npz"
    if not npz.exists():
        raise SystemExit(f"  no fingerprints for {args.episode}")
    data = np.load(npz)
    labels = cluster(data["vectors"],
                     json.loads((FINGERPRINTS / f"{args.episode}.json")
                                .read_text())["threshold"])
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
        full = json.loads(MAP.read_text()) if MAP.exists() else {}
        entry = full.setdefault(args.episode, {})
        for idx in by_cluster[which]:
            entry[str(idx)] = person
        MAP.write_text(json.dumps(full))
        print(f"  cluster {which} -> {person!r} on "
              f"{len(by_cluster[which])} lines of {episode['title'][:44]}")
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
        names = collections.Counter(known.get(str(i)) for i in idxs
                                    if known.get(str(i)))
        settled = names.most_common(1)[0][0] if names else None

        spread = idxs[::max(1, len(idxs) // SLICES)][:SLICES]
        starts = [segments[i]["t"] for i in spread if i < len(segments)]
        out = SAMPLES / f"{args.episode}-voice{which}.wav"
        ok = stitch(audio, starts, out)

        head = f"  voice {which}   {share*100:5.1f}% of the episode"
        print(head + (f"   — already named: {settled}" if settled else ""))
        if ok:
            print(f"     listen   afplay {out}")
        if starts:
            print(f"     in situ  {deep_link(episode, starts[0])}")
        sample = next((segments[i]["text"][:72] for i in spread
                       if i < len(segments) and segments[i]["text"].strip()),
                      "")
        print(f"     says     \"{sample}\"\n")

    print("  name one with:\n"
          f"    scripts/name_voices.py --episode {args.episode} "
          "--name <voice> \"Their Name\"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
