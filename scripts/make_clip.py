"""Cut a captioned clip from any indexed episode, on this machine.

    .venv/bin/python scripts/make_clip.py "what did luca netz say about pudgy penguins"
    .venv/bin/python scripts/make_clip.py --episode x-2075316750439338088 --at 4:01:47
    .venv/bin/python scripts/make_clip.py "..." --seconds 60 --height 720

Local rather than a web feature, deliberately. The hosted service runs on a
free tier with 0.1 CPU: ffmpeg encodes at about 2.6x realtime on this laptop
at full CPU, so the same work there would take minutes per clip AND block
every search request while it ran. Trading a search engine that works for a
clip button that might is the wrong trade. This gets the actual value — a
shareable clip of something nobody else has — at no hosting risk, and it
proves whether clips are worth hosting before anything is paid for.

Works for both sources. Clips are a convenience on either one, since a
viewer can reach the moment themselves: ?t=<seconds> seeks on YouTube and
on an X broadcast alike. What a clip from an X broadcast still gets you is
the material itself — about half of every live show never reaches the
upload, so for those minutes there is no YouTube copy to link to.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.clipper import (CLIP_HEIGHT,   # noqa: E402
    make_wide_overlay,  # noqa: E402
    build_captions,
    fetch_section,
    ffmpeg_available,
    make_backdrop,
    render,
    snap_to_speech,
    stamp,
)

# Both archives that carry transcripts. The Musk recordings were left out
# when this was written and stayed out, so every Musk post went up without
# a clip while the broadcast got one -- for no reason beyond a filename.
#
# MCG is deliberately absent: data/mcg_index.json is a shelf, not
# transcripts, and captions come from segments this machine does not hold.
EPISODES = [ROOT / "data" / "episodes.json",
            ROOT / "data" / "elon_episodes.json"]
SEARCH = "https://search.lexthedev.com"


# ─── MCG, and anything else with no transcript on this machine ────────────
#
# data/mcg_index.json is a shelf: titles, urls, durations. The transcripts
# live in Pinecone, and captions need per-line timings that a vector store
# does not hand back. So for those the captions come from YouTube's own
# auto-generated track, which is timed to the video and free to fetch.
#
# Not as good as the archive's Whisper text -- auto-subs punctuate badly
# and mishear names -- but a clip with slightly rough captions beats the
# 458 episodes that currently cannot be clipped at all.
def youtube_segments(url: str) -> list[dict]:
    """[{t, text}] from YouTube's auto-caption track."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sub"
        run = ["yt-dlp", "--skip-download", "--write-auto-sub",
               "--sub-lang", "en", "--sub-format", "vtt",
               "-o", str(out) + ".%(ext)s", url]
        subprocess.run(run, capture_output=True, text=True, timeout=180)
        found = list(Path(tmp).glob("*.vtt"))
        if not found:
            raise SystemExit("  youtube has no caption track for that video")
        return _parse_vtt(found[0].read_text())


_CUE = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d+)\s+-->")


def _parse_vtt(body: str) -> list[dict]:
    """VTT to segments, with the rolling duplication taken out.

    Auto-captions repeat: each cue re-prints the tail of the previous one
    so the words scroll. Kept verbatim that turns a 60-second clip into
    three minutes of stuttering subtitles, so only words not already
    carried forward are kept.
    """
    segments: list[dict] = []
    seen_tail: list[str] = []
    start = None
    for line in body.splitlines():
        cue = _CUE.search(line)
        if cue:
            h, m, sec, _ = cue.groups()
            start = int(h) * 3600 + int(m) * 60 + int(sec)
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()
        if start is None or not text or text.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        words = text.split()
        # Drop the longest prefix this cue shares with what came before.
        keep = words
        for cut in range(min(len(words), len(seen_tail)), 0, -1):
            if seen_tail[-cut:] == words[:cut]:
                keep = words[cut:]
                break
        if keep:
            segments.append({"t": float(start), "text": " ".join(keep)})
            seen_tail = (seen_tail + keep)[-40:]
    return segments


def parse_timestamp(value: str) -> float:
    """"4:01:47", "56:47" or "3407" -> seconds."""
    parts = [p for p in str(value).strip().split(":") if p != ""]
    if not parts:
        raise ValueError(f"could not read a timestamp from {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def top_hit(query: str) -> dict:
    """Ask the live search where the best moment for this question is."""
    request = urllib.request.Request(
        f"{SEARCH}/v1/podcast/search",
        data=json.dumps({"query": query}).encode(),
        headers={"content-type": "application/json"})
    body = json.loads(urllib.request.urlopen(request, timeout=180).read(),
                      strict=False)
    hits = body.get("hits") or []
    if not hits:
        sys.exit(f"no results for {query!r}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?",
                    help="a question; the top result becomes the clip")
    ap.add_argument("--episode", help="episode id, instead of a query")
    ap.add_argument("--youtube", metavar="URL",
                    help="any youtube video, captions taken from youtube "
                         "(for MCG and anything else not transcribed here)")
    ap.add_argument("--at", help="timestamp, e.g. 4:01:47 (with --episode)")
    ap.add_argument("--seconds", type=float, default=45.0,
                    help="clip length (default 45)")
    ap.add_argument("--lead", type=float, default=3.0,
                    help="seconds of run-up before the moment, so it does "
                         "not open mid-word (default 3)")
    # Defaults to whatever the website renders, so a clip cut here and a
    # clip a viewer cuts from the site are the same file. They were not: a
    # posted clip was 1080p and anyone who tried the button on the same
    # moment got 720p, which makes the tool look worse than the thing
    # advertising it. Pass --width to deliberately differ.
    ap.add_argument("--width", "--height", dest="width", type=int,
                    default=CLIP_HEIGHT, choices=[480, 720, 1080, 1280, 1920],
                    help=f"canvas width; the site uses {CLIP_HEIGHT}")
    # The broadcast is 1920 wide and the canvas is square, so a 1080
    # canvas scales it down to 1080 across and throws away nearly half
    # the horizontal detail before anything is even encoded. --best keeps
    # the picture at the size it arrives in and encodes for quality
    # rather than to a bitrate. Slower, larger, and the one to use for a
    # clip that is going out in public.
    # Highest quality and 16:9 are what a clip going out in public wants,
    # every time, so they are the default rather than something to
    # remember. The opt-outs exist for a quick look at a moment.
    ap.add_argument("--fast", action="store_true",
                    help="hardware encode at a fixed bitrate, for a look")
    # A square file pillarboxes in any 16:9 player and the black down both
    # sides is the first thing anyone notices.
    ap.add_argument("--square", action="store_true",
                    help="square canvas with the title in a band above")
    ap.add_argument("--title", help="overlay title, for --youtube")
    ap.add_argument("--out", help="output file (default: ~/Desktop)")
    args = ap.parse_args()
    args.best, args.wide = not args.fast, not args.square
    # There used to be a silent bump to 1920 here whenever --height was not
    # passed, which is how a posted clip came out 1080p while every viewer
    # cutting the same moment on the site got 720p. The default is now the
    # site's canvas, stated in --help, and differing takes an argument.

    if not ffmpeg_available():
        sys.exit("ffmpeg is not on PATH — brew install ffmpeg")

    episodes = {}
    for source in EPISODES:
        if source.exists():
            episodes.update({e["episode_id"]: e
                             for e in json.loads(source.read_text())})

    if args.youtube:
        if not args.at:
            sys.exit("  --youtube needs --at")
        vid = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", args.youtube)
        episode_id = vid.group(1) if vid else args.youtube
        episodes[episode_id] = {
            "episode_id": episode_id,
            "title": args.title or "MCG Live",
            "url": f"https://www.youtube.com/watch?v={episode_id}",
            "platform": "youtube",
            "segments": youtube_segments(args.youtube),
        }
        args.episode = episode_id

    if args.query:
        hit = top_hit(args.query)
        episode_id = hit["episode_id"]
        start = float(hit["start_seconds"])
        print(f"  top result: {hit['title'][:56]} @ {hit['timestamp']}")
    else:
        if not (args.episode and args.at):
            sys.exit("give a query, or --episode with --at")
        episode_id, start = args.episode, parse_timestamp(args.at)

    episode = episodes.get(episode_id)
    if episode is None:
        sys.exit(f"{episode_id} is in neither "
                 f"{' nor '.join(e.name for e in EPISODES)} "
                 "— re-fetch it first")

    # A clip that opens mid-syllable reads as broken, so back up a little.
    start = max(0.0, start - args.lead)
    end = start + args.seconds
    # ...and one that STOPS mid-syllable reads worse, because that is the
    # half a viewer is left on. The transcript knows where the sentences
    # are, so both edges move to where speech actually starts and stops.
    asked = end - start
    start, end = snap_to_speech(episode["segments"], start, end)
    if abs((end - start) - asked) > 0.05:
        print(f"  snapped to the sentence: {asked:.0f}s → {end - start:.0f}s")
    on_x = episode.get("platform") != "youtube"

    print(f"  {episode['title'][:60]}")
    print(f"  {stamp(start)} → {stamp(end)}  ({end - start:.0f}s, "
          f"{args.width}p, {'X broadcast' if on_x else 'YouTube'})")

    captions = build_captions(episode["segments"], start, end)
    out = Path(args.out) if args.out else (
        Path.home() / "Desktop" /
        f"clip-{episode_id}-{int(start)}s-{args.width}p.mp4")

    began = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        source = work / "src.mp4"
        print("  downloading the section…")
        fetch_section(episode["url"], start, end, source, height=args.width)

        backdrop = work / "backdrop.png"
        if args.wide:
            make_wide_overlay(episode["title"], stamp(start), backdrop,
                              args.width, int(args.width * 9 / 16))
        else:
            make_backdrop(episode["title"], stamp(start), backdrop,
                          args.width)

        print(f"  rendering {len(captions)} caption(s)…")
        render(source, captions, backdrop, work, out, args.width,
               best=args.best, wide=args.wide)

    size_mb = out.stat().st_size / 1_048_576
    print(f"\n  {out}")
    print(f"  {size_mb:.1f} MB in {time.time() - began:.0f}s")
    if on_x:
        # Worth saying out loud: this is the half of the show that the
        # YouTube upload cuts, and X cannot link to a timestamp, so the clip
        # is the only way to point anyone at this moment.
        print("  (from the live broadcast — this moment is not on YouTube, "
              "and X cannot link to a timestamp)")
    if shutil.which("open"):
        print("  open it:  open " + str(out).replace(" ", "\\ "))


if __name__ == "__main__":
    main()
