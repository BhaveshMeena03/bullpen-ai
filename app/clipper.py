"""Turn a moment into a downloadable video clip, on the server.

The page can already point at a moment and hand over a still card. What
people actually post is a short video with the words burned on, and asking
them to run a Python script is the same as not offering it.

Everything expensive about a clipper is already paid for here. The
transcript was indexed with timestamps months ago, so captions come out of
the index rather than out of a speech-to-text pass, and only the seconds
being clipped are ever downloaded — a minute of a three-hour episode is a
couple of megabytes, not the whole file.

What it costs, measured rather than guessed. A viewer clip renders at the
same quality as one cut by hand for a post — 1920 wide, 16:9, crf 16 — and
that moves both costs by roughly an order of magnitude:

  CPU      ~2.2 CPU-seconds per second of output. A 45s clip is ~100
           CPU-seconds: under a minute on two cores, four minutes on half
           of one. CPU is now a binding constraint, not a rounding error.
  egress   ~14MB for 45s, against 9MB for the old 720p square. Roughly 350
           clips per 5GB rather than 550.

Both are the price of the thing being worth posting. A 720p square
pillarboxes in every timeline and looks like a preview, and half this
archive is X broadcasts, which cannot be linked to a timestamp at all —
for those, this clip is the only way anyone can share the moment.

The limits below exist so a viral afternoon costs a queue and a refusal
rather than a bill:

  1920 wide, 16:9   what the source actually holds
  45s max           long enough for a real exchange
  1 at a time       encoding is the most expensive thing this service does,
                    and concurrency here is how a small box dies

Jobs run in the background and the client polls, because even a fast
encode is far longer than a request should be held open.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# The canvas is a parameter, not a constant. A clip made on a laptop should
# be the best the source allows; one made on a small server has to be the
# cheapest thing that still looks right on a phone. Same code, one number.
DEFAULT_SIZE = 720          # server default: half the bytes of 1080
BG = "#0b0e11"
GREEN = "#16c784"
CAPTION_WORDS = 5

# The credit on every clip. Short on purpose: this is read at a glance on a
# phone, off a re-upload, by somebody who has never heard of the site — the
# subdomain was three extra syllables that carried no information. It is the
# only attribution on a file that travels, which is why it is not optional.
SITE_CREDIT = "lexthedev.com"


# Three minutes is the ceiling, not the expectation. Measured end to end in
# a one-core container, a 20-second clip takes 91 seconds — download,
# caption render and encode — so this ceiling is about a quarter of an hour
# of work. The picker says so before the button is pressed, and
# MAX_CONCURRENT of 1 means the queue behind it is real. The default in the
# picker stays short; this is headroom for a whole exchange.
MAX_CLIP_SECONDS = 180
MIN_CLIP_SECONDS = 5

# What a viewer-requested clip renders at. 1920 wide, quality-targeted,
# same as a clip made by hand for a post — measured at ~2.2 CPU-seconds per
# second of output, so a 45s clip is ~100 CPU-seconds. That is a minute on
# two cores and four on half of one, which is why MAX_CONCURRENT is 1 and
# the front end shows a queue position rather than pretending it is instant.
CLIP_HEIGHT = 1920
# Only one encode at a time. The queue is what keeps a burst from turning
# into an out-of-memory kill on a small instance.
MAX_CONCURRENT = 1
# Finished clips are deleted after this. The disk is ephemeral anyway, and
# a clip nobody fetched in half an hour is one nobody wanted.
CLIP_TTL_SECONDS = 1800
MAX_JOBS_TRACKED = 200

# yt-dlp negotiates the media URL as one client and hands it to ffmpeg,
# which fetches it as itself; most clients tie the URL to the caller, so
# ffmpeg gets a 403 and the download fails as "ffmpeg exited with code 8".
# web_safari, web and tv refuse the format outright. mweb works — and that
# is the whole reason it was chosen, which turned out to be the bug.
#
# mweb is served ONE progressive format: 640x360. Not a preference among
# several, the only thing on offer. So every clip ever cut from a YouTube
# episode was sourced at 360p and then scaled UP to a 720 or 1080 canvas,
# and asking for a bigger canvas made it blurrier rather than sharper.
# Nothing in the pipeline could have recovered it; the detail was never
# downloaded.
#
# Empty means yt-dlp picks, and what it picks offers the real ladder up to
# 1920x1080 60fps. The 403 this was working around comes from handing a
# client-bound URL to ffmpeg, which only happens on the streaming path —
# the download here writes a file first, so it does not apply.
PLAYER_CLIENT = ""

# Fonts have to be found on both a Mac and a slim Debian image; neither
# has the other's. Missing fonts degrade to Pillow's bitmap default, which
# looks broken rather than plain, so the image installs DejaVu explicitly.
FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]
FONT_CANDIDATES_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]


def _font(candidates: list[str], size: int):
    from PIL import ImageFont
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    logger.warning("No TrueType font found; captions will look wrong.")
    return ImageFont.load_default()


def effective_cpus() -> int:
    """How many cores this process may actually use, not how many it can see.

    A container sees every core on the host and is scheduled on a fraction
    of one. x264 sizes its thread pool from the visible count, so on a 1-CPU
    box that reports 12 it starts twelve encoding threads, each with its own
    frame buffers, and they then timeshare a single core's worth of quota.
    Measured in exactly that shape: 2.07GB of anonymous memory against a 2GB
    limit, and a twenty-second clip still encoding after six minutes.

    The cgroup quota is the honest number, so it is read directly. Falls
    back to the visible count where there is no quota, which is the normal
    case on a laptop and correct there.
    """
    try:                                        # cgroup v2
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if raw[0] != "max":
            return max(1, round(int(raw[0]) / int(raw[1])))
    except Exception:                           # noqa: BLE001
        pass
    try:                                        # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0:
            return max(1, round(quota / period))
    except Exception:                           # noqa: BLE001
        pass
    return os.cpu_count() or 1


def _encoder(size: int = DEFAULT_SIZE, best: bool = False) -> list[str]:
    """Hardware encode on a Mac, x264 on the server.

    Checked once against the actual binary rather than assumed from the
    platform: the ffmpeg in the container is a different build from the
    one on a laptop, and guessing wrong fails at render time.

    Bitrate scales with the canvas. 2500k is right for 720 and visibly
    soft at 1080, and the only reason to hold 1080 down would be an egress
    bill that a local run does not have.

    `best` drops the hardware path entirely and encodes for quality rather
    than for a number of bits per second. VideoToolbox is fast and its
    fixed-bitrate mode is the weaker of the two ways to encode this: a
    static backdrop with a small moving panel spends its budget evenly
    whether the frame needs it or not, so the faces get what is left. x264
    at a constant quality spends bits where the picture is, and the render
    is a few seconds slower for something that gets re-encoded once by X
    and then watched. Only worth it for a clip that goes out publicly.
    """
    if best:
        # crf 16 and `slower`, not 18 and `slow`. X and YouTube both
        # re-encode whatever they are given, so this file is a source for
        # their encoder rather than the thing anyone watches — every
        # artefact left here is one their pass bakes in permanently.
        #
        # -tune film because that is what this footage is: real cameras and
        # grain, not flat animation. yuv420p and High@4.2 stay, since
        # anything more exotic will not decode in a phone timeline.
        return ["-c:v", "libx264", "-preset", "slower", "-crf", "16",
                "-tune", "film",
                # Sized to the quota, not to the core count the kernel
                # advertises. Without this the thread pool is built for the
                # host and starves on a container's slice.
                "-threads", str(effective_cpus()),
                "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2"]
    rate = f"{int(2500 * (size / DEFAULT_SIZE) ** 2)}k"
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        out = ""
    if "h264_videotoolbox" in out:
        return ["-c:v", "h264_videotoolbox", "-b:v", rate]
    return ["-c:v", "libx264", "-preset", "veryfast", "-b:v", rate,
            "-pix_fmt", "yuv420p"]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ─── captions ─────────────────────────────────────────────────────────────

def build_captions(segments: list[dict], start: float,
                   end: float) -> list[tuple[float, float, str]]:
    """Caption lines for the window, re-timed so 0 is the clip's start.

    The transcript arrives in ~8 second blocks, far too much text to read
    at once on a phone. Each block is split into short groups and the
    block's duration shared out by character count. An approximation, but
    the captions are auto-generated to begin with: legible and roughly in
    sync is the honest goal, not frame accuracy.
    """
    out: list[tuple[float, float, str]] = []
    window = [s for s in segments if start - 12 <= s.get("t", 0) < end]
    for i, seg in enumerate(window):
        seg_start = float(seg["t"])
        seg_end = float(window[i + 1]["t"]) if i + 1 < len(window) else seg_start + 8
        words = str(seg.get("text", "")).split()
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
                continue
            out.append((max(a, 0.0), min(b, end - start), chunk))
    return out


# ─── drawing ──────────────────────────────────────────────────────────────
#
# All text is drawn with Pillow and composited through `overlay`, rather
# than with ffmpeg's drawtext and subtitles filters. Those need libfreetype
# and libass, and the ffmpeg used while building this had neither, so they
# simply did not exist as filters. Depending on how someone compiled their
# ffmpeg is a bad foundation; overlay is in every build.

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


def _outlined(draw, xy, text, font, fill, outline=3):
    """Outline, not a caption box: a box is easier to read but covers the
    faces, and the faces are why a clip beats a screenshot."""
    x, y = xy
    for dx in range(-outline, outline + 1):
        for dy in range(-outline, outline + 1):
            if dx * dx + dy * dy <= outline * outline:
                draw.text((x + dx, y + dy), text, font=font, fill="black")
    draw.text((x, y), text, font=font, fill=fill)


def make_backdrop(title: str, stamp: str, path: Path,
                  size: int = DEFAULT_SIZE) -> None:
    from PIL import Image, ImageDraw

    k = size / DEFAULT_SIZE          # type scales with the canvas
    img = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size, int(7 * k)], fill=GREEN)

    font = _font(FONT_CANDIDATES_BOLD, int(27 * k))
    y = 34 * k
    for line in _wrap(draw, title, font, size - 90 * k)[:2]:
        w = draw.textlength(line, font=font)
        draw.text(((size - w) / 2, y), line, font=font, fill="#e6e8ea")
        y += 34 * k

    foot_font = _font(FONT_CANDIDATES_REGULAR, int(18 * k))
    stamp_font = _font(FONT_CANDIDATES_BOLD, int(18 * k))
    gap = 12 * k
    sw = draw.textlength(stamp, font=stamp_font)
    cw = draw.textlength(SITE_CREDIT, font=foot_font)
    x = (size - (sw + gap + cw)) / 2
    y = size - 44 * k
    draw.text((x, y), stamp, font=stamp_font, fill="#e6e8ea")
    draw.text((x + sw + gap, y), SITE_CREDIT, font=foot_font, fill=GREEN)
    img.save(path)


def make_wide_overlay(title: str, stamp: str, path: Path,
                      width: int = 1920, height: int = 1080) -> None:
    """Title and credit painted ON the picture, for a 16:9 clip.

    The square layout parks the video in a band and fills the space above
    and below it with a title and a credit. That reads well in a feed, and
    it means anything playing the file in a 16:9 window pillarboxes it —
    black down both sides, which is what people actually notice.

    Here the picture fills the frame and the text sits on top of it. The
    only furniture is a scrim behind each line, dark enough to read against
    a bright frame and short enough to leave the faces alone.
    """
    from PIL import Image, ImageDraw

    k = width / 1920                 # 1.0 at 1080p, and scales from there
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, int(7 * k)], fill=GREEN)

    # A gradient, not a flat band. At a flat 225 this covered the frame it
    # sat on — the broadcast's own Polymarket and Market Bubble marks are
    # right there in the top strip and vanished behind it, which reads as a
    # bar bolted over the video rather than part of it.
    #
    # Strongest along the top edge where the text sits and falling away to
    # nearly clear at the bottom of the band, so the picture comes through
    # the lower half while the type keeps something to sit on. The text is
    # outlined as well, which is what actually carries legibility here —
    # that is why the flat band could be dropped this far. It was raised to
    # 225 once because the broadcast's "LOS ANGELES 1:42 PM PT" read
    # through the credit; the outline solves that without the paint.
    band = int(74 * k)
    scrim = Image.new("RGBA", (width, band), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scrim)
    for row in range(band):
        # 170 at the top edge, 40 at the bottom.
        alpha = int(170 - (170 - 40) * (row / max(1, band - 1)))
        sd.line([(0, row), (width, row)], fill=(0, 0, 0, alpha))
    img.paste(scrim, (0, int(7 * k)), scrim)

    # Shrink to fit rather than crop. Taking the first wrapped line cut
    # "Why Ansem Thinks Ethereum Is Done.. | Market" and threw away the
    # episode number, which is the half a reader needs.
    foot_font = _font(FONT_CANDIDATES_REGULAR, int(24 * k))
    stamp_font = _font(FONT_CANDIDATES_BOLD, int(24 * k))
    gap = 14 * k
    credit_w = (draw.textlength(stamp, font=stamp_font) + gap
                + draw.textlength(SITE_CREDIT, font=foot_font))
    room = width - 68 * k - credit_w - 40 * k
    for pt in (36, 33, 30, 27, 24):
        font = _font(FONT_CANDIDATES_BOLD, int(pt * k))
        if draw.textlength(title, font=font) <= room:
            break
    shown = title
    while draw.textlength(shown, font=font) > room and len(shown) > 12:
        shown = shown[:-2]
    if shown != title:
        shown = shown.rstrip(" .|-") + "…"
    _outlined(draw, (34 * k, (74 * k - pt * k) / 2 + 7 * k), shown,
              font, "#e6e8ea", outline=max(2, int(3 * k)))

    # In the top band beside the title, not along the bottom. The
    # broadcast runs its own logo, chyron and ticker across the lower third
    # of every frame, so anything put down there is competing with three
    # things at once — and the captions have to live there too.
    # Two pieces, not one string. The timestamp is the evidence — bold and
    # white, the same weight as the title it sits beside — and the domain is
    # the credit, lighter and green so it reads as a mark rather than as
    # part of the sentence. One flat green run made them look like the same
    # fact, which they are not.
    baseline = (74 * k - 24 * k) / 2 + 7 * k
    ring = max(2, int(3 * k))
    x = width - credit_w - 34 * k
    _outlined(draw, (x, baseline), stamp, stamp_font, "#e6e8ea", outline=ring)
    x += draw.textlength(stamp, font=stamp_font) + gap
    _outlined(draw, (x, baseline), SITE_CREDIT, foot_font, GREEN, outline=ring)

    img.save(path)


def make_caption(text: str, path: Path, size: int = DEFAULT_SIZE,
                 height: int | None = None) -> None:
    from PIL import Image, ImageDraw

    k = size / DEFAULT_SIZE
    height = height or size
    # Type is sized against the SHORT edge, not the width. Scaling it off
    # the width put 93px lettering on a 1080-tall frame — nearly a tenth of
    # the picture, when a subtitle wants about a twentieth. On the square,
    # where the two are equal, nothing changes.
    kt = min(size, height) / DEFAULT_SIZE
    img = Image.new("RGBA", (size, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _font(FONT_CANDIDATES_BOLD, int(35 * kt))
    lines = _wrap(draw, text, font, size - 100 * k)[:3]
    # Measured up from the bottom of whatever frame this is. On the square
    # the gap under the picture is where the caption lives, so it is deep.
    # On a 16:9 frame the picture goes to the edge and the same gap would
    # park the words across the faces. They go to the very bottom instead,
    # over the broadcast's own ticker, which is the one strip of that
    # frame nothing is lost by covering.
    pad = (168 * k) if height >= size else (0.10 * height)
    step = 44 * kt
    y = height - pad - (len(lines) - 1) * step

    # A soft panel behind the words, only on the wide frame. There the
    # captions sit on the picture, and the broadcast's ticker underneath
    # them is bright, busy and moving — an outline alone has to fight it
    # every frame. On the square they sit on flat backdrop and need
    # nothing, which is what the outline was written for.
    if height < size and lines:
        widest = max(draw.textlength(ln, font=font) for ln in lines)
        bx, by = 34 * kt, 16 * kt
        panel = Image.new(
            "RGBA",
            (int(widest + bx * 2), int(len(lines) * step + by * 2)),
            (0, 0, 0, 110))
        img.paste(panel,
                  (int((size - widest) / 2 - bx), int(y - by)), panel)

    for line in lines:
        w = draw.textlength(line, font=font)
        _outlined(draw, ((size - w) / 2, y), line, font, "white",
                  outline=max(2, int(2.5 * kt)))
        y += step
    img.save(path)


# ─── the work ─────────────────────────────────────────────────────────────

def _ytdlp_binary() -> str:
    """Prefer the venv's yt-dlp over whatever is on PATH.

    In the container they are the same thing. Run from a script under a
    different interpreter they are not, and a bare "yt-dlp" raises
    FileNotFoundError — which is how moving this function out of the
    script and into the shared module broke the script.
    """
    local = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "yt-dlp"
    if local.exists():
        return str(local)
    return shutil.which("yt-dlp") or "yt-dlp"


def fetch_section(url: str, start: float, end: float, dest: Path,
                  proxy: str | None = None, height: int = DEFAULT_SIZE) -> None:
    """Download just the requested seconds of a video, at most `height` tall.

    Works for YouTube and for an X broadcast replay. The two need different
    handling in two places:

      player client   YouTube refuses a section download from some clients;
                      mweb is the one that works. X has no such argument and
                      rejects the whole flag, so it is only passed to
                      YouTube URLs.
      fragments       An X replay arrives as thousands of small HLS
                      fragments fetched one at a time by default, which is
                      bounded by round-trip latency rather than bandwidth.
                      Sixteen at once is the difference between seconds and
                      minutes even for a short section.
    """
    # No --force-keyframes-at-cuts. It makes yt-dlp re-encode the section
    # with x264 so the cut lands exactly on the requested frame — and this
    # pipeline then encodes the result a second time, so the work is paid
    # for twice and thrown away once. On a container with one core that is
    # the difference between a clip and a timeout: measured on the same
    # twenty-second section, 8 seconds without it against a 180-second
    # timeout with it.
    #
    # Accuracy was the reason it was there, so it was checked rather than
    # assumed: a 30-second request comes back 30.01 seconds long starting
    # at 0.03, which is well inside the tolerance for captions that are
    # timed from the requested start.
    is_youtube = "youtube.com" in url or "youtu.be" in url

    # --force-keyframes-at-cuts is per-source, because the two sources fail
    # in opposite directions.
    #
    # X is HLS: yt-dlp fetches the fragments itself, so the flag only buys a
    # frame-exact cut and costs an x264 re-encode this pipeline throws away.
    # Measured at 8 seconds without it against a 180-second timeout with it.
    #
    # YouTube hands ffmpeg a URL bound to the client that negotiated it
    # (c=ANDROID_VR), and ffmpeg fetches as itself and gets 403 — "ffmpeg
    # exited with code 8". With the flag, yt-dlp downloads the section
    # itself and ffmpeg never touches the network. So YouTube keeps it and
    # pays the re-encode; X does not.
    cmd = [_ytdlp_binary(), "--quiet", "--no-warnings",
           "--download-sections", f"*{start:.2f}-{end:.2f}",
           *(["--force-keyframes-at-cuts"] if is_youtube else []),
           # H.264 first so the merge stays an mp4. Left to itself yt-dlp
           # takes AV1 with Opus, which is a smaller download and a webm,
           # and then everything downstream is decoding AV1 for no benefit
           # — this gets re-encoded on the next line anyway.
           "-f", (f"bv*[height<={height}][vcodec^=avc1]+ba[ext=m4a]/"
                  f"bv*[height<={height}]+ba/b[height<={height}]/b"),
           "--concurrent-fragments", "16"]
    if is_youtube:
        if PLAYER_CLIENT:
            cmd += ["--extractor-args",
                    f"youtube:player_client={PLAYER_CLIENT}"]
        cmd += ["--remote-components", "ejs:github"]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd += ["-o", str(dest), url]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not dest.exists():
        tail = ((result.stderr or "").strip().splitlines() or ["(no stderr)"])[-1]
        raise RuntimeError(f"could not fetch that section: {tail[:180]}")


def render(source: Path, captions, backdrop: Path, workdir: Path,
           out: Path, size: int = DEFAULT_SIZE, best: bool = False,
           wide: bool = False) -> None:
    """Compose the clip.

    Square by default: the picture sits in a band with the title above it
    and the credit below. `wide` fills a 16:9 frame with the picture and
    lays the same text over it, which is the layout to use when the clip
    will be watched rather than scrolled past — a square file pillarboxes
    in any 16:9 player, and the black down both sides is the first thing
    anyone notices.
    """
    height = int(size * 9 / 16) if wide else size
    inputs = ["-i", str(source), "-loop", "1", "-i", str(backdrop)]
    if wide:
        # Cover, not fit: fill the frame and crop the overflow rather than
        # leaving a bar. The source is already 16:9, so this crops nothing
        # in practice and protects the frame if one ever is not.
        steps = [f"[0:v]scale={size}:{height}:force_original_aspect_ratio="
                 f"increase,crop={size}:{height}[vid]",
                 "[vid][1:v]overlay=0:0:shortest=1[base]"]
    else:
        steps = [f"[0:v]scale={size}:-2[vid]",
                 "[1:v][vid]overlay=(W-w)/2:(H-h)/2:shortest=1[base]"]
    label = "base"
    for i, (start, end, text) in enumerate(captions):
        png = workdir / f"cap{i:04d}.png"
        make_caption(text, png, size, height)
        inputs += ["-i", str(png)]
        nxt = f"c{i}"
        steps.append(f"[{label}][{i + 2}:v]overlay=0:0:"
                     f"enable='between(t,{start:.2f},{end:.2f})'[{nxt}]")
        label = nxt

    result = subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(steps),
         "-map", f"[{label}]", "-map", "0:a?", *_encoder(size, best),
         "-c:a", "aac", "-b:a", "256k" if best else "128k",
         "-movflags", "+faststart",
         "-shortest", str(out)],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        tail = "; ".join((result.stderr or "").strip().splitlines()[-3:])
        raise RuntimeError(f"render failed: {tail[:220]}")


def stamp(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ─── jobs ─────────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | working | done | failed
    error: str | None = None
    path: Path | None = None
    created: float = field(default_factory=time.monotonic)
    title: str = ""
    seconds: float = 0.0


class ClipService:
    """Background clip jobs, one at a time, with everything bounded."""

    def __init__(self, proxy: str | None = None):
        self._jobs: dict[str, Job] = {}
        self._gate = asyncio.Semaphore(MAX_CONCURRENT)
        self._proxy = proxy
        self._dir = Path(tempfile.gettempdir()) / "clips"
        self._dir.mkdir(exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    def _sweep(self) -> None:
        now = time.monotonic()
        for job_id, job in list(self._jobs.items()):
            if now - job.created > CLIP_TTL_SECONDS:
                if job.path and job.path.exists():
                    job.path.unlink(missing_ok=True)
                self._jobs.pop(job_id, None)
        while len(self._jobs) > MAX_JOBS_TRACKED:
            oldest = min(self._jobs.values(), key=lambda j: j.created)
            if oldest.path:
                oldest.path.unlink(missing_ok=True)
            self._jobs.pop(oldest.id, None)

    def get(self, job_id: str) -> Job | None:
        self._sweep()
        return self._jobs.get(job_id)

    def queued_count(self) -> int:
        return sum(1 for j in self._jobs.values()
                   if j.status in ("queued", "working"))

    # -- the job -----------------------------------------------------------

    def submit(self, episode: dict, start: float, end: float) -> Job:
        self._sweep()
        job = Job(id=uuid.uuid4().hex[:12], title=episode.get("title", ""),
                  seconds=end - start)
        self._jobs[job.id] = job
        asyncio.create_task(self._run(job, episode, start, end))
        return job

    async def _run(self, job: Job, episode: dict,
                   start: float, end: float) -> None:
        async with self._gate:
            job.status = "working"
            try:
                out = self._dir / f"{job.id}.mp4"
                # Blocking work: yt-dlp and ffmpeg are subprocesses, and
                # running them inline would stall the event loop for every
                # other request on this single-worker server.
                await asyncio.to_thread(self._build, episode, start, end, out)
                job.path = out
                job.status = "done"
                logger.info("clip %s done (%.0fs, %.1fMB)", job.id,
                            end - start, out.stat().st_size / 1e6)
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.error = str(exc)[:200]
                logger.warning("clip %s failed: %s", job.id, job.error)

    def _build(self, episode: dict, start: float,
               end: float, out: Path) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            raw = work / "raw.mp4"
            # Same recipe make_clip.py uses for a clip that goes out in
            # public: full height, 16:9, quality-targeted encode. The old
            # defaults here were 720 square at a fixed bitrate, which is a
            # preview — and a viewer sharing this has no other way to point
            # anyone at an X broadcast, since X cannot link to a timestamp.
            fetch_section(episode["url"], start, end, raw, self._proxy,
                          height=CLIP_HEIGHT)
            backdrop = work / "backdrop.png"
            make_wide_overlay(episode.get("title", ""), stamp(start),
                              backdrop, CLIP_HEIGHT,
                              int(CLIP_HEIGHT * 9 / 16))
            captions = build_captions(episode.get("segments", []), start, end)
            render(raw, captions, backdrop, work, out, CLIP_HEIGHT,
                   best=True, wide=True)
