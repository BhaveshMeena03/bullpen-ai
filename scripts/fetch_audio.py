"""Re-download the audio for every indexed episode, for diarization.

    .venv/bin/python scripts/fetch_audio.py
    .venv/bin/python scripts/fetch_audio.py --only F4OhqZjtVkY
    .venv/bin/python scripts/fetch_audio.py --list

The transcription pipeline downloads audio, uses it, and deletes it — the
transcript was the only thing worth keeping, so data/ is 10MB of JSON for
86 hours of talking. Diarization needs the audio back: who is speaking is
a fact about the sound, and it is not recoverable from the words.

Resumable on purpose. This is 33 downloads over hours, some of them
four-hour broadcasts arriving as ten thousand HLS fragments, and any of
them can fail on a socket. Anything already on disk is skipped, so
re-running after a failure costs only what is missing.

The files land in audio/, which is gitignored and roughly 40MB an hour —
about 3.5GB for the archive. Delete it when the labels are built; the
transcripts and the speaker map are the things worth keeping.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EPISODES = ROOT / "data" / "episodes.json"
AUDIO = ROOT / "audio"


def _bin(name: str) -> str:
    local = ROOT / ".venv" / "bin" / name
    return str(local) if local.exists() else (shutil.which(name) or name)


def download(url: str, dest: Path) -> bool:
    """One episode's audio as 16 kHz mono mp3.

    Format string copied from transcribe_x_broadcast, and for the reason
    documented there: an X broadcast has no audio-only rendition, so
    "bestaudio/best" quietly falls through to `best` and pulls gigabytes
    of video for a five-hour show. Whisper and every diarizer resample to
    16 kHz mono anyway, so the smallest rendition loses nothing.
    """
    # Sixteen at once turns hours into minutes, and on some streams the
    # fragments race each other: "No such file or directory:
    # ...mp4.part-Frag68", reproducibly, on two of thirty-three episodes.
    # Serial is slow and always works, so it is the retry rather than the
    # default.
    for fragments in ("16", "1"):
        # Leftover fragment state makes the retry resume a download that no
        # longer exists, and fail differently.
        for stale in dest.parent.glob(f"{dest.name}.*"):
            if stale.suffix != ".mp3":
                stale.unlink(missing_ok=True)
        try:
            subprocess.run(
                # --no-progress: the progress bar redraws thousands of times
                # per download and wrote 3.6MB of carriage returns into the
                # log of a single episode, burying the line that mattered.
                [_bin("yt-dlp"), "--socket-timeout", "30", "--no-warnings",
                 "--no-progress", "--concurrent-fragments", fragments,
                 "-f", "worstaudio/worstvideo+bestaudio/worst",
                 "--extract-audio", "--audio-format", "mp3",
                 "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000 -b:a 32k",
                 "-o", str(dest.with_suffix(".%(ext)s")), url],
                check=True, timeout=7200,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if fragments == "16":
                print(f"     {type(exc).__name__} — retrying one fragment "
                      f"at a time")
                continue
            print(f"     failed: {type(exc).__name__}")
            return False
        return dest.with_suffix(".mp3").exists()
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[], metavar="EPISODE_ID")
    ap.add_argument("--list", action="store_true",
                    help="show what is present and what is missing, and stop")
    args = ap.parse_args()

    episodes = json.loads(EPISODES.read_text())
    if args.only:
        wanted = set(args.only)
        episodes = [e for e in episodes if e["episode_id"] in wanted]

    AUDIO.mkdir(exist_ok=True)
    have = [e for e in episodes
            if (AUDIO / f"{e['episode_id']}.mp3").exists()]
    missing = [e for e in episodes if e not in have]

    if args.list:
        size = sum((AUDIO / f"{e['episode_id']}.mp3").stat().st_size
                   for e in have) / 1e9
        print(f"\n  {len(have)}/{len(episodes)} downloaded ({size:.1f} GB)")
        for e in missing:
            print(f"     missing  {e.get('title', '?')[:56]}")
        print()
        return 0

    # Smallest first. An X broadcast has no audio-only rendition, so its
    # smallest option is still ~900MB of video for a four hour show, while
    # a YouTube episode is a few tens of MB. Doing the cheap ones first
    # means a failure three hours in has not cost everything.
    missing.sort(key=lambda e: e["segments"][-1]["t"] if e.get("segments") else 0)
    print(f"\n  {len(have)} already downloaded, {len(missing)} to fetch")
    print("  (smallest first; X broadcasts pull video because they have no "
          "audio-only stream)\n")
    began = time.time()
    failed = []
    for i, episode in enumerate(missing, 1):
        title = episode.get("title", "?")[:52]
        print(f"  [{i}/{len(missing)}] {title}", flush=True)
        dest = AUDIO / episode["episode_id"]
        if not download(episode["url"], dest):
            failed.append(episode)

    done = len(missing) - len(failed)
    print(f"\n  {done}/{len(missing)} fetched in {(time.time()-began)/60:.0f} min")
    if failed:
        # Named, not just counted: a rerun should be one command and not a
        # hunt through scrollback.
        print(f"  {len(failed)} failed — rerun to retry just these:")
        for e in failed:
            print(f"     --only {e['episode_id']}   {e.get('title','?')[:44]}")
    total = sum(f.stat().st_size for f in AUDIO.glob("*.mp3")) / 1e9
    print(f"  audio/ is {total:.1f} GB\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
