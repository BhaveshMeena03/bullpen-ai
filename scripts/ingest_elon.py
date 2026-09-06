"""Transcribe the verified Elon Musk sources into their own archive.

    .venv/bin/python scripts/ingest_elon.py            # everything pending
    .venv/bin/python scripts/ingest_elon.py --only dEv99vxKjVI
    .venv/bin/python scripts/ingest_elon.py --list

Writes data/elon_episodes.json. Deliberately not episodes.json: the two
archives never mix. @mbubbleSearch's whole standing is that it answers
from the Market Bubble broadcast, and an answer about that show sourced
from a Tesla interview would end that in one reply.

Only interviews. The Tesla earnings calls have the best provenance in the
whole list -- official channel, quarterly, dated by the title -- and they
are left out, because attribution fails on them. Voice clustering put 66%
of a call into one cluster that would not split at any number of clusters,
and the transcript's own hand-offs ("Thanks Lars", "our next question is
coming from Walt") disagreed with the clusters about who was speaking. On
an archive whose entire claim is "Elon said this", a quarter's guidance
from the CFO attributed to him is the failure that ends the project.

The same test on a two-person interview split 48/44 with the clusters
matching the speakers by content -- Lex's third-person introduction in
one, Elon answering in the first person in the other. That is the shape
the pipeline already handles on Market Bubble, so that is what is ingested.

Resumable. A 14-hour batch is hours of laptop time and something will
interrupt it; anything already transcribed is skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.episode_store import load as load_episodes, merge  # noqa: E402
from app.provenance import admissible, drop_hallucinated   # noqa: E402

OUT = ROOT / "data" / "elon_episodes.json"
AUDIO_DIR = Path("/tmp/elon_audio")
# The project's own yt-dlp when there is one, otherwise whatever is on
# PATH. Hardcoding the venv path meant this script could only run from a
# checkout that had one: in a worktree it failed with "No such file or
# directory: .venv/bin/yt-dlp", which reads like a download failure and
# is not one. Episode #49 looked like a YouTube problem for a day because
# of it.
YTDLP = next((str(p) for p in (ROOT / ".venv" / "bin" / "yt-dlp",
                               Path.home() / "bullpen-concierge" / ".venv"
                               / "bin" / "yt-dlp")
              if p.is_file()), shutil.which("yt-dlp") or "yt-dlp")
COOKIES = Path.home() / "Downloads" / "cookies.txt"

# Verified by hand against the collector's output: every one is on the
# channel that recorded it. Adding to this list means checking that first.
SOURCES = [
    ("dEv99vxKjVI", "Lex Fridman", "2019-04-12",
     "Elon Musk: Tesla Autopilot | Lex Fridman Podcast #18"),
    ("smK9dgdTl40", "Lex Fridman", "2019-11-12",
     "Elon Musk: Neuralink, AI, Autopilot, and the Pale Blue Dot | "
     "Lex Fridman Podcast #49"),
    ("DxREm3s1scA", "Lex Fridman", "2021-12-28",
     "Elon Musk: SpaceX, Mars, Tesla Autopilot, Self-Driving, Robotics, "
     "and AI | Lex Fridman Podcast #252"),
    ("JN3KPFbWCy8", "Lex Fridman", "2023-11-09",
     "Elon Musk: War, AI, Aliens, Politics, Physics, Video Games, and "
     "Humanity | Lex Fridman Podcast #400"),
    ("Kbk9BiPhm7o", "Lex Fridman", "2024-08-02",
     "Elon Musk: Neuralink and the Future of Humanity | "
     "Lex Fridman Podcast #438"),
]


def load() -> list[dict]:
    return load_episodes(OUT)


def fetch_audio(video_id: str) -> Path:
    AUDIO_DIR.mkdir(exist_ok=True)
    path = AUDIO_DIR / f"{video_id}.m4a"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    # Clients rotate per attempt, and the format selector falls back rather
    # than insisting. "bestaudio" alone is not always offered -- episode #49
    # refused it outright with "Requested format is not available" -- and
    # which client answers depends on what YouTube is challenging that day.
    # Same lesson as the clipper: vary the client, not just the retry.
    jar = None
    if COOKIES.is_file():
        import shutil
        jar = Path("/tmp/yt-cookies-ingest.txt")
        shutil.copyfile(COOKIES, jar)

    last = ""
    for client in ("tv_embedded", "", "web_embedded", "android"):
        cmd = [YTDLP, "--no-warnings",
               # Progressive audio, then any audio, then whatever exists.
               "-f", "bestaudio[ext=m4a]/bestaudio/best",
               "--extract-audio", "--audio-format", "m4a",
               "-o", str(path)]
        if client:
            cmd += ["--extractor-args", f"youtube:player_client={client}"]
        if jar:
            cmd += ["--cookies", str(jar)]
        cmd += [f"https://www.youtube.com/watch?v={video_id}"]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if done.returncode == 0 and path.exists():
            return path
        last = ((done.stderr or "").strip().splitlines() or ["?"])[-1]
        path.unlink(missing_ok=True)
    raise RuntimeError(f"download failed: {last[:160]}")


def transcribe(path: Path) -> list[dict]:
    import mlx_whisper
    result = mlx_whisper.transcribe(
        str(path), path_or_hf_repo="mlx-community/whisper-turbo",
        language="en", verbose=False)
    return [{"t": round(s["start"], 2), "text": s["text"].strip()}
            for s in result.get("segments", []) if s.get("text", "").strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="one video id")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    have = {e["episode_id"] for e in load()}
    todo = [s for s in SOURCES
            if (not args.only or s[0] == args.only) and s[0] not in have]

    if args.list:
        for vid, channel, date, title in SOURCES:
            mark = "done" if vid in have else "    "
            print(f"  {mark}  {date}  {vid}  {title[:56]}")
        return 0

    print(f"  {len(have)} already in {OUT.name}, {len(todo)} to do\n")
    for vid, channel, date, title in todo:
        ok, why = admissible(channel, title)
        if not ok:
            print(f"  REFUSED {vid}: {why}")
            continue
        print(f"  {vid}  {title[:54]}", flush=True)
        try:
            audio = fetch_audio(vid)
            print(f"     {audio.stat().st_size // 1_000_000} MB, "
                  f"transcribing…", flush=True)
            segments = transcribe(audio)
        except Exception as exc:                            # noqa: BLE001
            print(f"     failed: {exc}")
            continue
        kept, dropped = drop_hallucinated(segments)
        if dropped:
            print(f"     dropped hallucinated: {', '.join(dropped)}")
        # Through the store, which takes an exclusive lock and re-reads
        # inside it. This runs for hours and something else may be writing;
        # a plain write_text here would silently drop whatever landed
        # between this script's last read and its next save.
        merge([{
            "episode_id": vid,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "platform": "youtube",
            "published_at": date,
            "channel": channel,
            "segments": kept,
        }], path=OUT)
        hours = max((s["t"] for s in kept), default=0) / 3600
        print(f"     {len(kept)} segments, {hours:.1f}h -> {OUT.name}\n",
              flush=True)

    total = load()
    hours = sum(max((s["t"] for s in e["segments"]), default=0)
                for e in total) / 3600
    print(f"  archive: {len(total)} episodes, {hours:.1f} hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
