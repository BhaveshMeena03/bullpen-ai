"""Cut a shareable video clip from a moment in the podcast.

Search finds the moment; this turns it into something postable. The static
card the web page offers is a quote on a background — fine for a link, but
nobody stops scrolling for it. What gets watched is a short video with the
words burned on, which is what the people clipping this show already make
by hand.

The advantage here is that the hard part is already done. A clipper
normally has to transcribe the audio before it can caption anything; this
repo indexed the transcript months ago, with timestamps, so the captions
come out of data/episodes.json for free and line up with the speech
because they came from it.

    # find a moment, then cut it
    .venv/bin/python scripts/make_clip.py --query "why gamers make the best traders"
    .venv/bin/python scripts/make_clip.py --query "..." --pick 1 --duration 45

    # or cut a known moment directly
    .venv/bin/python scripts/make_clip.py --episode TYX2FuacIhE --start 2662

Output is a 1080x1080 MP4 in clips/ — square, because it takes the most
feed height on X without being cropped.

Runs locally, not on the server. Fetching and re-encoding video needs real
CPU and moves hundreds of megabytes; the free Render instance has neither
the cores nor the bandwidth, and the residential proxy is billed per
gigabyte. A laptop does it in well under a minute.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "episodes.json"
OUT_DIR = ROOT / "clips"
API = "https://search.lexthedev.com"

# Square. Portrait wins on TikTok, but this is going to X, where 1:1 shows
# the most pixels in a timeline without the crop that 9:16 gets.
SIZE = 1080
BG = "#0b0e11"          # same as the site, so a clip reads as the same product
GREEN = "#16c784"

# Caption chunking. The transcript arrives in ~8s blocks, which is far too
# much text to read at once on a phone. Splitting into short groups and
# spreading the block's duration across them by character count is an
# approximation, but the captions are auto-generated to begin with — the
# honest goal is legible and roughly in sync, not frame-accurate.
CAPTION_WORDS = 5


def _ytdlp() -> str:
    local = ROOT / ".venv" / "bin" / "yt-dlp"
    return str(local) if local.exists() else "yt-dlp"


def _proxy_args() -> list[str]:
    import os
    proxy = os.environ.get("YTDLP_PROXY", "").strip()
    return ["--proxy", proxy] if proxy else []


def load_episodes() -> dict:
    if not DATA.exists():
        sys.exit(f"{DATA} not found — run scripts/fetch_episodes.py first.")
    return {e["episode_id"]: e for e in json.loads(DATA.read_text())}


def find_moments(query: str, top_k: int = 5) -> list[dict]:
    """Ask the live search where the moment is. Semantic, so the query can
    be how you remember it rather than what was said."""
    r = httpx.post(f"{API}/v1/podcast/search",
                   json={"query": query, "top_k": top_k}, timeout=120)
    r.raise_for_status()
    return r.json().get("hits", [])


# ─── captions ─────────────────────────────────────────────────────────────

def build_captions(segments: list[dict], start: float,
                   end: float) -> list[tuple[float, float, str]]:
    """Captions for the window, re-timed so 0 is the start of the clip."""
    out: list[tuple[float, float, str]] = []
    window = [s for s in segments if start - 12 <= s["t"] < end]
    for i, seg in enumerate(window):
        seg_start = seg["t"]
        seg_end = window[i + 1]["t"] if i + 1 < len(window) else seg_start + 8
        words = seg["text"].split()
        if not words:
            continue
        chunks = [" ".join(words[j:j + CAPTION_WORDS])
                  for j in range(0, len(words), CAPTION_WORDS)]
        total = max(sum(len(c) for c in chunks), 1)
        cursor = seg_start
        for chunk in chunks:
            share = (seg_end - seg_start) * (len(chunk) / total)
            a, b = cursor - start, cursor + share - start
            cursor += share
            if b <= 0 or a >= end - start:
                continue                      # outside the clip
            out.append((max(a, 0.0), min(b, end - start), chunk))
    return out


# ─── text, drawn with Pillow rather than ffmpeg ───────────────────────────
#
# Every piece of text here is rendered to a PNG and composited, instead of
# using ffmpeg's drawtext and subtitles filters. Not a stylistic choice: the
# ffmpeg on this machine is built without libfreetype and libass, so neither
# filter exists, and a clipper that only runs on a correctly-compiled ffmpeg
# is one `brew upgrade` away from breaking. Pillow renders identically
# wherever it is installed, and overlay is in every build.

FONT_DIRS = [
    "/System/Library/Fonts/Supplemental/",
    "/System/Library/Fonts/",
    "/Library/Fonts/",
]
BOLD_CANDIDATES = ["Arial Bold.ttf", "Helvetica.ttc", "HelveticaNeue.ttc",
                   "Arial.ttf", "Supplemental/Arial Bold.ttf"]
REGULAR_CANDIDATES = ["Arial.ttf", "Helvetica.ttc", "HelveticaNeue.ttc"]


def _font(candidates: list[str], size: int):
    from PIL import ImageFont
    for directory in FONT_DIRS:
        for name in candidates:
            path = Path(directory) / name
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size)
                except OSError:
                    continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) > max_width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _outlined(draw, xy, text, font, fill, outline=4):
    """Text with a hard outline. Legible over any frame, and unlike a solid
    caption box it does not cover the faces, which are the reason a clip
    beats a screenshot."""
    x, y = xy
    for dx in range(-outline, outline + 1):
        for dy in range(-outline, outline + 1):
            if dx * dx + dy * dy <= outline * outline:
                draw.text((x + dx, y + dy), text, font=font, fill="black")
    draw.text((x, y), text, font=font, fill=fill)


def make_backdrop(title: str, stamp: str, path: Path) -> None:
    """The parts that never change: background, rule, title, footer."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, SIZE, 10], fill=GREEN)

    title_font = _font(BOLD_CANDIDATES, 40)
    lines = _wrap(draw, title, title_font, SIZE - 130)[:2]
    y = 52
    for line in lines:
        w = draw.textlength(line, font=title_font)
        draw.text(((SIZE - w) / 2, y), line, font=title_font, fill="#e6e8ea")
        y += 50

    foot_font = _font(REGULAR_CANDIDATES, 27)
    foot = f"{stamp}   ·   search.lexthedev.com"
    w = draw.textlength(foot, font=foot_font)
    draw.text(((SIZE - w) / 2, SIZE - 66), foot, font=foot_font, fill=GREEN)
    img.save(path)


def make_caption(text: str, path: Path) -> None:
    """One caption, full-canvas and transparent, so ffmpeg overlays it at
    0:0 and the positioning lives here rather than in the filter string."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _font(BOLD_CANDIDATES, 52)
    lines = _wrap(draw, text, font, SIZE - 150)[:3]
    y = SIZE - 250 - (len(lines) - 1) * 60
    for line in lines:
        w = draw.textlength(line, font=font)
        _outlined(draw, ((SIZE - w) / 2, y), line, font, "white")
        y += 60
    img.save(path)


def render(source: Path, captions: list[tuple[float, float, str]],
           backdrop: Path, workdir: Path, out: Path) -> None:
    """Composite: video on the backdrop, captions timed over the top."""
    inputs = ["-i", str(source), "-loop", "1", "-i", str(backdrop)]
    steps = [f"[0:v]scale={SIZE}:-2[vid]",
             f"[1:v][vid]overlay=(W-w)/2:(H-h)/2:shortest=1[base]"]

    label = "base"
    for i, (start, end, text) in enumerate(captions):
        png = workdir / f"cap{i:04d}.png"
        make_caption(text, png)
        inputs += ["-i", str(png)]
        nxt = f"c{i}"
        steps.append(
            f"[{label}][{i + 2}:v]overlay=0:0:"
            f"enable='between(t,{start:.2f},{end:.2f})'[{nxt}]")
        label = nxt

    result = subprocess.run(
        ["ffmpeg", "-y", *inputs,
         "-filter_complex", ";".join(steps),
         "-map", f"[{label}]", "-map", "0:a?",
         # videotoolbox: hardware encode on Apple silicon, several times
         # faster than libx264 and more than good enough for a social clip.
         "-c:v", "h264_videotoolbox", "-b:v", "6M",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
         "-shortest", str(out)],
        capture_output=True, text=True)
    if result.returncode != 0:
        # Surfacing this matters: the first version captured stderr and
        # raised a CalledProcessError, so a missing filter showed up as a
        # bare exit code and cost a debugging round to identify.
        tail = "\n    ".join((result.stderr or "").strip().splitlines()[-4:])
        sys.exit(f"ffmpeg failed:\n    {tail}")


# yt-dlp negotiates the media URL as one client and then hands that URL to
# ffmpeg, which fetches it as itself. Most YouTube clients tie the URL to
# the client that asked for it, so ffmpeg gets a 403 and the download dies
# with "ffmpeg exited with code 8" and no useful explanation. Of the
# clients tried (web_safari, web, tv all refuse the format outright), mweb
# is the one that both offers progressive formats and issues a URL ffmpeg
# can actually read.
PLAYER_CLIENT = "mweb"


def fetch_section(url: str, start: float, end: float, dest: Path,
                  height: int = 1080) -> None:
    """Download only the window, not the whole three-hour episode."""
    cmd = [_ytdlp(), "--quiet", "--no-warnings",
           "--download-sections", f"*{start:.2f}-{end:.2f}",
           "--force-keyframes-at-cuts",
           "-f", f"bv*[height<={height}]+ba/b[height<={height}]/b",
           "--extractor-args", f"youtube:player_client={PLAYER_CLIENT}",
           "--remote-components", "ejs:github",
           *_proxy_args(), "-o", str(dest), url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dest.exists():
        tail = (result.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
        sys.exit(f"could not fetch that section: {tail[0][:200]}")


def timestamp(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def parse_time(value: str) -> float:
    """Accept 1:02:03, 44:22 or plain seconds.

    The timestamps on the page and in the search results are written the
    first way, so requiring seconds would mean doing arithmetic to use the
    thing you are looking at.
    """
    value = str(value).strip()
    if ":" not in value:
        return float(value)
    parts = [float(p) for p in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"can't read {value!r} as a time")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="find the moment by meaning")
    ap.add_argument("--pick", type=int, help="which search result to cut (1-based)")
    ap.add_argument("--episode", help="episode id, if you already know it")
    # Times take 44:22 or 1:02:03 or raw seconds, so a timestamp copied off
    # the page or a search result can be pasted in as it appears.
    ap.add_argument("--start", help="where the clip begins, e.g. 44:22")
    ap.add_argument("--end", help="where it ends, e.g. 45:30 (overrides --duration)")
    ap.add_argument("--duration", type=float, default=50.0,
                    help="clip length; X plays up to 2:20, but 40-60s holds")
    ap.add_argument("--lead", type=float, default=3.0,
                    help="seconds before the moment, so it doesn't open mid-word "
                         "(ignored when --start is given: that is your call)")
    ap.add_argument("--height", type=int, default=1080,
                    help="source resolution ceiling")
    args = ap.parse_args()

    episodes = load_episodes()

    if args.query and not args.episode:
        hits = find_moments(args.query)
        if not hits:
            sys.exit("no moments found for that query.")
        if not args.pick:
            print(f"\n  moments for {args.query!r}:\n")
            for i, h in enumerate(hits, 1):
                text = re.sub(r"\s+", " ", h["text"])[:96]
                print(f"   {i}. {h['timestamp']:>8}  {h['title'][:52]}")
                print(f"      {text}…\n")
            print("  re-run with --pick N to cut one.\n")
            return 0
        hit = hits[args.pick - 1]
        episode_id = hit["episode_id"]
        # Search suggests where the moment is; --start overrides it, because
        # a retrieval window begins where a chunk boundary fell, not where
        # the thought starts. Only the person watching knows that.
        found_at = float(hit["start_seconds"])
        start = parse_time(args.start) if args.start else max(0.0, found_at - args.lead)
    elif args.episode and args.start is not None:
        episode_id, start = args.episode, parse_time(args.start)
    else:
        sys.exit("give --query (then --pick N), or --episode with --start.")

    episode = episodes.get(episode_id)
    if not episode:
        sys.exit(f"{episode_id} is not in data/episodes.json.")

    end = parse_time(args.end) if args.end else start + args.duration
    if end <= start:
        sys.exit(f"--end ({timestamp(end)}) must come after --start ({timestamp(start)}).")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{episode_id}-{int(start)}.mp4"

    print(f"  episode : {episode['title'][:66]}")
    print(f"  window  : {timestamp(start)} → {timestamp(end)}  ({end - start:.0f}s)")

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        print("  fetching the section …", flush=True)
        fetch_section(episode["url"], start, end, raw, args.height)

        captions = build_captions(episode["segments"], start, end)
        print(f"  captions: {len(captions)} lines drawn from the transcript")

        backdrop = Path(tmp) / "backdrop.png"
        make_backdrop(episode["title"], timestamp(start), backdrop)

        print("  rendering …", flush=True)
        render(raw, captions, backdrop, Path(tmp), out)

    size = out.stat().st_size / 1_000_000
    print(f"\n  {out}  ({size:.1f} MB)")
    print(f"  {episode['url']}&t={int(start)}s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
