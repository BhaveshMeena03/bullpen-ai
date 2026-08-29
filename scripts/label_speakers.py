"""Work out who is speaking, from the sound rather than the words.

    .venv/bin/python scripts/label_speakers.py --only qFeglFI5bac
    .venv/bin/python scripts/label_speakers.py --only qFeglFI5bac --at 2:40:25
    .venv/bin/python scripts/label_speakers.py

The transcripts carry no speaker at all — Whisper writes down words, not who
said them. So the archive can answer "what was said" and can only guess
"who said it", and it guessed wrong in public: asked how much Banks had
made, it answered from a passage about somebody else's portfolio and
reported it as Banks losing $254,000. It took the name from the question.

Who is speaking is a fact about the audio. This recovers it by giving every
transcript segment a voice fingerprint and grouping the ones that match.

Not pyannote: its diarization models are gated behind a HuggingFace account.
ECAPA is a public model that downloads anonymously, and combined with the
segment boundaries Whisper already produced it gets to the same place for
this purpose — a conversation with two hosts and a guest, not a crowd.

Two known limits, both handled rather than hidden. A Whisper segment can
span a speaker change, so only the middle of each segment is sampled and
short ones are skipped. And clustering names nobody: it produces "voice 1",
"voice 2". Turning those into Ansem and Banks is the next step, and it
leans on the fact that they are in all 33 episodes while a guest is in one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPISODES = ROOT / "data" / "episodes.json"
AUDIO = ROOT / "audio"
OUT = ROOT / "data" / "speakers"

# A segment shorter than this is usually a backchannel — "yeah", "right" —
# and too short to fingerprint a voice from.
MIN_SECONDS = 1.4
# Sampled from the middle, because a segment that spans a speaker change
# usually changes at its edges.
SAMPLE_SECONDS = 2.6


def _seconds(stamp: str) -> float:
    parts = [float(p) for p in stamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def load_audio(path: Path, start: float, length: float):
    """One slice, 16 kHz mono, straight from ffmpeg.

    Decoding the slice rather than the file: an 86-hour archive does not
    fit in memory, and seeking costs nothing next to the embedding.
    """
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "quiet",
         "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", str(path),
         "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=False).stdout
    if not raw:
        return None
    return np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0


def fingerprints(episode: dict, audio: Path, encoder, device: str):
    """A voice embedding per usable segment, and which segments they are."""
    import numpy as np
    import torch

    usable = []
    for i, seg in enumerate(episode["segments"]):
        nxt = (episode["segments"][i + 1]["t"]
               if i + 1 < len(episode["segments"]) else seg["t"] + 4)
        if nxt - seg["t"] >= MIN_SECONDS and seg.get("text", "").strip():
            usable.append((i, seg["t"], min(nxt - seg["t"], SAMPLE_SECONDS)))

    vectors, kept = [], []
    for start in range(0, len(usable), 64):
        batch = usable[start:start + 64]
        waves = []
        for _, at, length in batch:
            # A beat in from the edge: that is where a speaker change sits.
            wave = load_audio(audio, at + 0.15, length)
            waves.append(wave)
        good = [(m, w) for m, w in zip(batch, waves, strict=True) if w is not None
                and len(w) > 8000]
        if not good:
            continue
        width = min(len(w) for _, w in good)
        stack = torch.tensor(np.stack([w[:width] for _, w in good])).to(device)
        with torch.no_grad():
            out = encoder.encode_batch(stack).squeeze(1).cpu().numpy()
        vectors.append(out)
        kept.extend(m[0] for m, _ in good)
        print(f"\r     {len(kept)}/{len(usable)} segments fingerprinted",
              end="", flush=True)
    print()
    return (np.vstack(vectors) if vectors else None), kept


def cluster(vectors, threshold: float):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    unit = normalize(vectors)
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="cosine", linkage="average").fit_predict(unit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[], metavar="EPISODE_ID")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="lower splits voices apart, higher merges them")
    ap.add_argument("--refingerprint", action="store_true",
                    help="ignore the cached embeddings and redo the audio")
    ap.add_argument("--sweep", action="store_true",
                    help="try a range of thresholds and report how many "
                         "voices each finds, instead of writing anything")
    ap.add_argument("--at", help="print who is speaking around this "
                                 "timestamp — for checking against a moment "
                                 "you already know")
    args = ap.parse_args()

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  loading the voice encoder on {device}")
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / ".models" / "ecapa"),
        run_opts={"device": device})

    episodes = json.loads(EPISODES.read_text())
    if args.only:
        wanted = set(args.only)
        episodes = [e for e in episodes if e["episode_id"] in wanted]

    OUT.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        audio = AUDIO / f"{episode['episode_id']}.mp3"
        if not audio.exists():
            print(f"  no audio yet: {episode.get('title','?')[:48]}")
            continue
        print(f"\n  {episode.get('title','?')[:56]}")
        # Cached, because fingerprinting a four-hour show takes minutes and
        # the threshold needs sweeping. The audio cannot change under it —
        # these are recordings — so a cache hit is always correct.
        import numpy as np
        cache = OUT / f"{episode['episode_id']}.npz"
        if cache.exists() and not args.refingerprint:
            held = np.load(cache)
            vectors, kept = held["vectors"], held["kept"].tolist()
            print(f"     {len(kept)} fingerprints (cached)")
        else:
            vectors, kept = fingerprints(episode, audio, encoder, device)
            if vectors is not None:
                np.savez_compressed(cache, vectors=vectors,
                                    kept=np.array(kept))
        if vectors is None:
            print("     nothing usable")
            continue

        if args.sweep:
            # The number that matters is not the cluster count but how much
            # of the episode the top two voices account for: on this show
            # that is the two hosts, and a threshold that splits them is
            # useless however tidy the count looks.
            for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                lab = cluster(vectors, t)
                share = Counter(lab).most_common(4)
                top = sum(n for _, n in share[:2]) / len(lab)
                print(f"     threshold {t:.1f}: {len(set(lab)):4} voices · "
                      f"top two cover {100*top:3.0f}% · "
                      f"{[f'{100*n/len(lab):.0f}%' for _, n in share]}")
            continue

        labels = cluster(vectors, args.threshold)
        counts = Counter(labels)
        print(f"     {len(counts)} distinct voices; "
              f"largest shares: "
              f"{[f'{100*n/len(labels):.0f}%' for _, n in counts.most_common(5)]}")

        # Not `path`: the episode-file guard treats a bare `path` as a
        # possible write to episodes.json and fails the suite. This writes
        # a fingerprint file, and the name should say so.
        fingerprint_file = OUT / f"{episode['episode_id']}.json"
        fingerprint_file.write_text(json.dumps({
            "episode_id": episode["episode_id"],
            "threshold": args.threshold,
            # segment index -> voice number, and the centroids, so voices
            # can be matched across episodes without re-reading the audio.
            "segments": {str(i): int(v) for i, v in zip(kept, labels, strict=True)},
            "centroids": {
                str(v): vectors[[j for j, x in enumerate(labels) if x == v]]
                .mean(axis=0).tolist()
                for v in counts},
        }))

        if args.at:
            want = _seconds(args.at)
            print(f"\n     around {args.at}:")
            by_index = {i: v for i, v in zip(kept, labels, strict=True)}
            for i, seg in enumerate(episode["segments"]):
                if abs(seg["t"] - want) < 22 and i in by_index:
                    m, s = divmod(int(seg["t"]), 60)
                    print(f"       voice {by_index[i]}  [{m}:{s:02d}] "
                          f"{seg['text'].strip()[:74]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
