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
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .dedupe import episode_number

logger = logging.getLogger(__name__)

# The canvas is a parameter, not a constant. A clip made on a laptop should
# be the best the source allows; one made on a small server has to be the
# cheapest thing that still looks right on a phone. Same code, one number.
DEFAULT_SIZE = 720          # server default: half the bytes of 1080
BG = "#0b0e11"
GREEN = "#16c784"
CAPTION_WORDS = 5

# Clips used to carry a "lexthedev.com" credit, on the reasoning that a
# file which travels should say where it came from. Removed on purpose:
# these are posted alongside the show's own moments, and a URL burned into
# somebody else's broadcast reads as branding their footage rather than
# citing it. The episode and the timestamp stay, which is what a viewer
# actually needs to go and check the quote.


# Two minutes is the ceiling, not the expectation. Memory no longer scales
# with length, so this is purely about the wait: measured end to end in a
# one-core container a 20-second clip takes 91 seconds, which puts the
# ceiling at roughly ten minutes of work with a queue of one behind it.
# Three minutes was headroom nobody was using at fifteen minutes a go. The
# picker says the estimate before the button is pressed.
MAX_CLIP_SECONDS = 120
MIN_CLIP_SECONDS = 5

# What a viewer-requested clip renders at. 1280 wide, quality-targeted.
#
# It was 1920, and 1920 is why search went down on 4 September. Measured,
# each in a clean process, rendering the same 180-second clip:
#
#   1920x1080   1616 MB peak      9.6 MB file
#   1280x720     759 MB peak      5.6 MB file
#
# The render runs inside the web process on a 2 GB instance that is already
# holding the app and the episode cache, so 1616 MB is not a tight fit, it
# is an out-of-memory kill that takes search with it. 759 MB fits with room.
#
# Nothing about the encode changed: still crf 16, still quality-targeted,
# still the canvas the source actually fills. X and YouTube re-encode
# whatever they are handed, and 720p is what most of that timeline is
# watched at anyway. A clip cut by hand can still ask for 1920 by passing
# the size in, which is what the local path does.
CLIP_HEIGHT = 1280
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
# So the clients are tried in order rather than chosen once, and only
# clients that are offered the full ladder are in the list.
#
# Measured against a real episode, counting formats and the tallest on
# offer:
#
#   tv_embedded    43 formats, 1080p     <- first choice
#   default        43 formats, 1080p
#   web_embedded   29 formats, 1080p
#   android        1 format,   360p      <- excluded
#   mweb           1 format,   360p      <- excluded
#   ios            0 formats            <- excluded
#   tv             refuses outright     <- excluded
#
# android and mweb are left out on purpose even though they succeed. A
# 360p source scaled up to a 1080x1920 canvas is a visibly soft clip going
# out under this account's name, and the whole reason this renders at CRF
# 16 is that it is a source for someone else's re-encode. Failing honestly
# is better than shipping that quietly.
#
# tv_embedded leads because it is the client least likely to be met with
# "Sign in to confirm you're not a bot", which is what a residential proxy
# IP now draws from YouTube once the exit-IP mismatch is fixed. Empty
# string means no extractor-args at all, which is yt-dlp's own default.
PLAYER_CLIENTS = ("tv_embedded", "", "web_embedded")

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


# ─── where to cut ─────────────────────────────────────────────────────────

# A sentence that has actually finished, rather than a line that happens
# to have ended. Trailing quotes and brackets are allowed after the stop
# because captions carry them: `he said "we're done."` ends a sentence.
_SENTENCE_END = re.compile(r"""[.!?]['")\]]*\s*$""")

# The same stop, found anywhere in a line rather than only at its end.
# Requires whitespace or the line's end after it, so "3.5" and "u.s." do
# not read as the end of a thought.
_SENTENCE_STOP = re.compile(r"""[.!?]['")\]]*(?=\s|$)""")

# How far the cut may move to find a better place to land. Six seconds is
# about two spoken lines: enough to reach the end of a sentence in normal
# speech, short enough that a 45-second clip is still a 45-second clip.
SNAP_SLACK = 6.0
# How far back the opening may reach for the end of the previous sentence.
# Wider than SNAP_SLACK because the two edges fail differently: an over-long
# tail runs past the point, while extra lead-in is just a beat of context
# before the speaker starts the thought.
LEAD_SLACK = 10.0


def snap_to_speech(segments: list[dict], start: float, end: float,
                   slack: float = SNAP_SLACK) -> tuple[float, float]:
    """Move a clip's edges to where speech starts and stops.

    A clip cut at `start + duration` lands wherever the arithmetic puts
    it, which is usually the middle of a word. The moment is right and the
    clip reads as broken -- and a broken-sounding clip is worse than no
    clip, because it is the thing that gets posted.

    The transcript already knows where speech begins and ends: every
    segment carries the second it starts. So the end moves to a segment
    boundary near the requested one, preferring a boundary where the line
    before it finished a sentence; the start moves back to the beginning
    of whatever segment it landed inside.

    Both edges only ever move outward or to a boundary, never inward past
    the moment being clipped -- the point is to include the whole thought,
    not to trim it. Returns the original values unchanged when there is
    nothing better within `slack`, so a clip is never made worse.
    """
    if not segments:
        return start, end

    starts = [float(s.get("t", 0.0)) for s in segments]

    # Open at the top of the segment the start fell inside, so the first
    # word is whole. Never move forward: that would clip the moment.
    opening = [t for t in starts if t <= start]
    if opening and start - opening[-1] <= slack:
        start = opening[-1]

    # Where do sentences actually finish? Not at segment boundaries --
    # Whisper breaks a segment when it has heard enough audio, not when
    # the speaker has finished a thought, so on real transcripts the
    # boundaries land mid-clause ("a good trade right n-", "does have
    # currentl-"). Cutting only on them leaves the clip hanging exactly
    # the way it did before.
    #
    # The stops are inside the text instead, so each one is timed by how
    # far through the line it falls. That is an estimate, and it is the
    # same estimate build_captions already makes to time caption lines.
    candidates: list[tuple[float, bool]] = []
    for i, t in enumerate(starts):
        text = segments[i].get("text", "") or ""
        span = (starts[i + 1] - t) if i + 1 < len(starts) else 4.0
        if text and span > 0:
            for stop in _SENTENCE_STOP.finditer(text):
                through = (stop.end() / len(text))
                candidates.append((t + through * span, True))
        candidates.append((t, False))          # the boundary, as a fallback

    # The start gets the same treatment the end already had, and for the
    # same reason. Snapping it to a segment top makes the first WORD whole;
    # it does not make the first SENTENCE whole, so clips still opened
    # mid-thought ("...and that's why I sold it"), which is the one flaw
    # that makes a clip unpostable and can only be found by watching it.
    #
    # A sentence ends here, so the speech just after it begins one. Only
    # ever moved EARLIER: moving the start forward onto a later boundary
    # would trim the front off the moment being clipped. The cost of being
    # wrong is therefore a couple of seconds of lead-in, which is why the
    # window is wider than the end's.
    lead_in = [when for when, finished in candidates
               if finished and start - LEAD_SLACK <= when <= start]
    if lead_in:
        start = round(max(lead_in), 2)

    best, best_cost = None, None
    for when, finished in candidates:
        if abs(when - end) > slack or when <= start:
            continue
        # A finished sentence is worth up to the whole slack window; among
        # equals, the cut nearest the requested length wins.
        cost = abs(when - end) - (slack if finished else 0.0)
        if best_cost is None or cost < best_cost:
            best, best_cost = when, cost
    if best is not None:
        end = round(best, 2)
    return start, end


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

    stamp_font = _font(FONT_CANDIDATES_BOLD, int(18 * k))
    sw = draw.textlength(stamp, font=stamp_font)
    x = (size - sw) / 2
    y = size - 44 * k
    draw.text((x, y), stamp, font=stamp_font, fill="#e6e8ea")
    img.save(path)


def short_title(title: str) -> str:
    """The show and the episode number, and nothing else.

    The full title is "LIVE W/ WILL CLEMENTE, NET NET CAPITAL, & TYLER
    BERNABE: Market Bubble Ep 18 - Presented by @Polymarket". Painted
    across the top of the frame it ran straight through the broadcast's own
    Polymarket wordmark and neither could be read. A guest list is also not
    what a viewer needs on a thirty-second clip; the caption carries that.
    """
    number = episode_number(title or "")
    if number:
        return f"Market Bubble Ep {number}"
    plain = (title or "").split(" - Presented")[0].split(":")[-1].strip()
    return (plain[:30].rstrip(" .|-") + "…") if len(plain) > 30 else plain


def make_wide_overlay(title: str, stamp: str, path: Path,
                      width: int = 1920, height: int = 1080) -> None:
    """Title and credit painted ON the picture, for a 16:9 clip.

    Everything sits in the top-RIGHT corner, and that is the whole design.
    The broadcast puts its Polymarket wordmark top-left, its own Market
    Bubble logo and a sponsor chyron bottom-left, and a scrolling ticker
    across the very bottom. The top right is the one corner of the frame it
    leaves alone.

    It used to be a full-width band with the title along the left, which
    covered the Polymarket mark on every clip -- the show's sponsor,
    obscured by a tool that exists to serve the show. A clip should look
    like it came from the broadcast, not like something was bolted over it.

    The scrim is only as wide as the text needs, so the picture is
    untouched everywhere else, and the type is outlined, which is what
    actually carries legibility over a moving frame.
    """
    from PIL import Image, ImageDraw

    k = width / 1920                 # 1.0 at 1080p, and scales from there
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # The one full-width mark that stays: a hairline, not a bar.
    draw.rectangle([0, 0, width, int(5 * k)], fill=GREEN)

    label = short_title(title)
    title_font = _font(FONT_CANDIDATES_BOLD, int(25 * k))
    stamp_font = _font(FONT_CANDIDATES_BOLD, int(25 * k))
    dot_font = _font(FONT_CANDIDATES_REGULAR, int(25 * k))

    dot = "  ·  "
    parts = [(label, title_font, "#e6e8ea"),
             (dot, dot_font, "#8a939e"),
             (stamp, stamp_font, "#e6e8ea")]
    run = sum(draw.textlength(t, font=f) for t, f, _ in parts)

    # Tucked up under the hairline rather than floating below it. The
    # broadcast puts its own "LOS ANGELES / 1:59 PM PT" chyron in this
    # same black band, starting about 48px down at 1080p, and the panel
    # used to sit at 19..66 -- straight through it, our title crossing
    # their clock. Everything above 48 is empty in every episode, so the
    # panel is raised and its padding tightened to fit inside it.
    pad_x, pad_y = 20 * k, 8 * k
    top = 5 * k + 2 * k
    box_h = 25 * k + pad_y * 2
    box_w = run + pad_x * 2
    left = width - box_w - 28 * k

    # A soft rounded panel behind just this line. Wide enough to read
    # against a bright frame, small enough that the rest of the picture is
    # exactly what the broadcast showed.
    panel = Image.new("RGBA", (int(box_w), int(box_h)), (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle(
        [0, 0, int(box_w) - 1, int(box_h) - 1], radius=int(10 * k),
        fill=(0, 0, 0, 105))
    img.paste(panel, (int(left), int(top)), panel)

    x = left + pad_x
    baseline = top + pad_y
    ring = max(2, int(3 * k))
    for text, font, colour in parts:
        _outlined(draw, (x, baseline), text, font, colour, outline=ring)
        x += draw.textlength(text, font=font)

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


# A dropped HLS fragment is the failure that does not announce itself. An X
# replay arrives as thousands of two-second fragments fetched sixteen at a
# time through a rotating proxy; when one of them quietly does not arrive,
# yt-dlp still exits 0 and still writes a playable file. The file is simply
# missing two seconds of picture while keeping all of its audio, so the clip
# renders, uploads, and plays with the video running two seconds behind the
# voices. That shipped, and only looked wrong to a person watching it.
#
# Fragments are about two seconds, so anything approaching that is a lost
# one rather than a slow frame.
FRAME_GAP_TOLERANCE = 0.5


def largest_frame_gap(path: Path) -> float:
    """The biggest hole between consecutive video frames, in seconds.

    Zero when the file cannot be probed: this guards a download, and a
    broken probe should not be able to reject a good one.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120).stdout
    except Exception:                                       # noqa: BLE001
        return 0.0
    stamps = []
    for chunk in out.replace(",", " ").split():
        try:
            stamps.append(float(chunk))
        except ValueError:
            continue
    if len(stamps) < 2:
        return 0.0
    return max(b - a for a, b in zip(stamps, stamps[1:]))



def pin_one_exit_ip(proxy: str, session: str) -> str:
    """The same proxy URL, asking for one IP rather than any IP.

    A rotating residential proxy hands out a different exit per
    connection. yt-dlp negotiates the media URL from one of them and
    ffmpeg, a separate process, fetches it from another, so YouTube sees
    the URL used from an address it was not issued to and refuses:
    "ffmpeg exited with code 8", which is a 403 wearing a different
    number. Retrying could not help, because every attempt split the same
    way.

    DataImpulse pins an exit by putting a session id in the username --
    login__sessid.abc:password@gw.dataimpulse.com:823 -- and holds that IP
    for about thirty minutes. Both processes read the same proxy string,
    so putting it there is what makes them share an address.

    Parameters are appended to the username after "__" and separated with
    ";", so an existing "__cr.us" is extended rather than replaced. A
    proxy that already names a session is left exactly as it is.
    """
    if not proxy or "sessid." in proxy:
        return proxy
    parsed = urlsplit(proxy)
    if not parsed.username:
        return proxy                    # no credentials to hang it off
    user = parsed.username
    user += f";sessid.{session}" if "__" in user else f"__sessid.{session}"
    auth = f"{user}:{parsed.password}@" if parsed.password else f"{user}@"
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, auth + host, parsed.path,
                       parsed.query, parsed.fragment))


def fetch_section(url: str, start: float, end: float, dest: Path,
                  proxy: str | None = None, height: int = DEFAULT_SIZE,
                  cookies: str | None = None) -> None:
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
    # A signed-in session is the one thing that answers "Sign in to confirm
    # you're not a bot". The caller hands over a jar yt-dlp may write to;
    # see _writable_cookie_jar.
    if cookies and Path(cookies).is_file():
        cmd += ["--cookies", cookies]
    cmd += ["-o", str(dest), url]
    # ffmpeg has to go through the proxy too, not just yt-dlp.
    #
    # --download-sections hands the range fetch to ffmpeg, which is a
    # separate process and knows nothing about --proxy. So yt-dlp
    # negotiates a media URL from the proxy's IP, ffmpeg then requests it
    # from this host's IP, and YouTube rejects the mismatch — "ffmpeg
    # exited with code 8", which is a 403 wearing a different number.
    # Setting the proxy on the child environment is what ffmpeg reads.
    # Built per attempt now, since each one asks for its own exit IP.
    # Retry, because a residential proxy hands out a different exit IP every
    # time and a share of that pool is already flagged by YouTube. A single
    # attempt is a coin flip on which IP you draw — observed directly: the
    # same credentials failed with 407 NO_USER and then passed a minute
    # later from a different exit. Three attempts turn a ~60% draw into
    # ~94%, and each one is free apart from the wait.
    #
    # Three attempts whether or not a proxy is set: a dropped fragment is a
    # network event, and the direct path drops them too, just less often.
    #
    # On YouTube each attempt also changes the player client, so two
    # independent things vary per try rather than one. Fixing the exit-IP
    # mismatch turned "ffmpeg exited with code 8" into "Sign in to confirm
    # you're not a bot", which is YouTube challenging the residential IP
    # itself: a different address alone does not answer that, a different
    # client can. Every client in the list is offered the full ladder, so
    # falling back costs nothing in quality.
    #
    # One YouTube attempt deliberately goes direct, with no proxy at all.
    # Nobody has ever tested whether YouTube actually refuses this host's
    # own address: the proxy was added for the exit-IP mismatch, which is
    # fixed, and it has since produced three different failures in three
    # runs -- a 403, a bot challenge, and a proxy intercepting TLS with a
    # self-signed certificate. Residential exits are not uniform, and a
    # datacentre address that simply works would be faster, free, and
    # nothing to keep alive. If it is refused, that costs one quick failure
    # and the log finally says so.
    clients = PLAYER_CLIENTS if is_youtube else ("",)
    attempts = max(3, len(clients))
    # Every attempt's outcome, not just the last one's. The job only ever
    # surfaced the final error, so three different things could go wrong in
    # one run and the report named one of them -- which is how four runs
    # produced four errors and no idea which attempt each belonged to.
    trail: list[str] = []
    direct_attempt = 2 if (is_youtube and proxy) else None
    last = ""
    for attempt in range(1, attempts + 1):
        # One exit IP per attempt, shared by both processes. A new session
        # each time, so a retry still draws a different address -- which is
        # the whole reason retrying works -- while yt-dlp and ffmpeg inside
        # one attempt agree on which address that is.
        env, run = None, list(cmd)
        client = clients[(attempt - 1) % len(clients)]
        if client:
            run = [*run[:-3], "--extractor-args",
                   f"youtube:player_client={client}", *run[-3:]]
        # --remote-components fetches a JavaScript runtime from GitHub at
        # request time so yt-dlp can solve YouTube's player challenge. When
        # that fetch is blocked or slow the extractor comes back with no
        # formats at all, which surfaces as "Requested format is not
        # available" -- the error this has been failing with, and one that
        # reads like a format-selector problem rather than a network one.
        # The last attempt drops it so the trail can tell the two apart.
        if is_youtube and attempt < attempts:
            run = [*run[:-3], "--remote-components", "ejs:github", *run[-3:]]
        direct = attempt == direct_attempt
        if proxy and not direct:
            pinned = pin_one_exit_ip(proxy, uuid.uuid4().hex[:12])
            run = [*run[:-3], "--proxy", pinned, *run[-3:]]
            env = {**os.environ,
                   "http_proxy": pinned, "https_proxy": pinned,
                   "HTTP_PROXY": pinned, "HTTPS_PROXY": pinned}
        elif direct:
            # The inherited environment may carry proxy variables of its
            # own, and this attempt is only meaningful if nothing routes.
            env = {k: v for k, v in os.environ.items()
                   if k.lower() not in ("http_proxy", "https_proxy",
                                        "all_proxy")}
        result = subprocess.run(run, capture_output=True, text=True,
                                timeout=180, env=env)
        if result.returncode == 0 and dest.exists():
            gap = largest_frame_gap(dest)
            if gap <= FRAME_GAP_TOLERANCE:
                if attempt > 1:
                    # Which client worked is the only way to learn what
                    # this host actually needs; the answer differs between
                    # a laptop on a home connection and a container behind
                    # a residential proxy, and only the second one matters.
                    logger.info("fetched on attempt %d of %d (client %s, "
                                "%s)", attempt, attempts,
                                client or "default",
                                "direct" if direct else "via proxy")
                return
            # Exit code 0 and a playable file, with a hole in it.
            last = (f"the download is missing {gap:.1f}s of video "
                    f"(a dropped fragment)")
            logger.warning("attempt %d came back with a %.1fs gap — "
                           "discarding it", attempt, gap)
        else:
            last = ((result.stderr or "").strip().splitlines()
                    or ["(no stderr)"])[-1]
        trail.append(f"{client or 'default'}"
                     f"{' direct' if direct else ''}"
                     f"{' no-ejs' if is_youtube and attempt == attempts else ''}"
                     f": {last[:110]}")
        # A partial file from the failed attempt would make the next one
        # look like it succeeded.
        dest.unlink(missing_ok=True)
        if attempt < attempts:
            logger.info("fetch attempt %d failed on client %s (%s) — %s",
                        attempt, client or "default", last[:90],
                        "was direct, next tries the proxy" if direct
                        else "retrying on a new exit IP")
    raise RuntimeError("could not fetch that section — "
                       + " | ".join(trail))


def _blank_frame(path: Path, size: int, height: int) -> None:
    """A fully transparent frame, for the stretches with nothing to say."""
    from PIL import Image
    Image.new("RGBA", (size, height), (0, 0, 0, 0)).save(path)


def _concat_list(captions, workdir: Path, size: int, height: int,
                 blank: Path) -> str:
    """A concat script covering the whole clip, gaps included.

    The demuxer plays entries back to back with no notion of a timeline, so
    silence has to be written down as the blank frame rather than left out.
    Without it every caption would slide earlier by the length of the gap
    before it and the whole track would drift out of sync.

    The last entry is repeated without a duration because the demuxer drops
    the final file's frame otherwise, which loses the closing caption.
    """
    lines: list[str] = []
    cursor = 0.0
    last: Path | None = None
    for i, (start, end, text) in enumerate(captions):
        if start - cursor > 0.04:               # a gap worth writing down
            lines.append(f"file '{blank.name}'")
            lines.append(f"duration {start - cursor:.3f}")
            last = blank
        png = workdir / f"cap{i:04d}.png"
        make_caption(text, png, size, height)
        held = max(end - max(start, cursor), 0.04)
        lines.append(f"file '{png.name}'")
        lines.append(f"duration {held:.3f}")
        last = png
        cursor = max(end, cursor)
    if last is not None:
        lines.append(f"file '{last.name}'")
    return "\n".join(lines) + "\n"


# How much memory one render may use before it is killed, in megabytes.
#
# The render is the largest thing this service does and it runs inside the
# web process's container. On 4 September a 1920x1080 clip peaked at 1.7GB
# on a 2GB instance and the kernel killed the container, which took search
# down with it -- a clip nobody was waiting for cost every search request
# that minute.
#
# A ceiling turns that into a failed clip. ffmpeg's allocation fails, it
# exits non-zero, the job reports an error, and search never notices.
# Measured peaks to size it against: 846MB at 720p, 1730MB at 1080p.
#
# Zero disables it, which is right for a laptop with memory to spare.
# DISABLED, because 1400 was measured against the wrong number and broke
# every clip in production within twenty minutes of shipping.
#
# `ulimit -v` bounds VIRTUAL address space. The peaks this was sized
# against -- 846MB at 720p, 1730MB at 1080p -- are RESIDENT memory, and
# ffmpeg reserves far more address space than it ever makes resident, so a
# 720p render that comfortably fits in RAM was refused an allocation and
# died. The failure was mine and the mechanism was wrong, not the idea.
#
# Re-enabling it needs a virtual-size figure measured on Linux, not a
# resident one measured on a laptop:
#
#   /usr/bin/time -v ffmpeg ...   → "Maximum resident set size" is not it;
#   read VmPeak from /proc/<pid>/status while a render runs.
#
# Until that number exists this stays off, and an over-budget render can
# still take the container down. That is the state it was in before, and a
# broken clipper is worse than a rare crash.
CLIP_MEMORY_LIMIT_MB = 0


def run_watching_memory(cmd: list[str], timeout: int,
                        env: dict | None = None):
    """Run a command and report the peak VIRTUAL size it reached, in MB.

    This exists because the memory ceiling was set from the wrong number
    and broke every clip. `ulimit -v` bounds virtual address space; the
    figures it was sized against were resident set sizes measured on a
    laptop, and ffmpeg reserves far more address space than it ever makes
    resident. The gap between those two numbers is the bug, and nothing on
    a Mac can measure it -- VmPeak is a Linux file.

    So the render measures itself, in the place that matters, and the log
    carries the answer. Polled rather than read once at the end, because
    /proc/<pid> disappears the moment the process does.

    Returns (completed_process, peak_mb). peak_mb is None where it cannot
    be measured, which is everywhere except Linux.
    """
    if not sys.platform.startswith("linux"):
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env), None

    import threading
    peak = {"kb": 0}

    def watch(pid: int, stop: threading.Event) -> None:
        status = Path(f"/proc/{pid}/status")
        while not stop.wait(0.4):
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("VmPeak:"):
                        peak["kb"] = max(peak["kb"], int(line.split()[1]))
                        break
            except (OSError, ValueError, IndexError):
                return                      # the process ended; keep the max

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    stop = threading.Event()
    watcher = threading.Thread(target=watch, args=(proc.pid, stop), daemon=True)
    watcher.start()
    try:
        out, err = proc.communicate(timeout=timeout)
    finally:
        stop.set()
    done = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    return done, (peak["kb"] / 1024 if peak["kb"] else None)


def _looks_like_out_of_memory(returncode: int, stderr: str) -> bool:
    """Whether ffmpeg died against the ceiling rather than on the input."""
    if returncode in (137, -9):                 # SIGKILL, the kernel's OOM
        return True
    lowered = (stderr or "").lower()
    return any(mark in lowered for mark in
               ("cannot allocate memory", "out of memory",
                "error allocating", "std::bad_alloc", "killed"))


def _capped(cmd: list[str], limit_mb: int = CLIP_MEMORY_LIMIT_MB) -> list[str]:
    """The same command, unable to use more than `limit_mb` of memory.

    Wrapped in a shell running `ulimit -v` rather than passed through
    subprocess's preexec_fn: preexec_fn runs between fork and exec in a
    process that has threads, which is documented as unsafe and this render
    is called from a thread pool. `exec "$@"` hands the arguments on
    without going back through shell quoting, so a filename with a space in
    it cannot become two arguments.

    Linux only. macOS counts mapped address space very differently and a
    limit that is generous there still refuses allocations ffmpeg makes
    routinely, so a developer machine is left alone.
    """
    if not limit_mb or not sys.platform.startswith("linux"):
        return cmd
    return ["/bin/sh", "-c", f'ulimit -v {limit_mb * 1024}; exec "$@"',
            "sh", *cmd]


def _has_audio(source: Path) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(source)],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return bool(out)
    except (OSError, subprocess.SubprocessError):
        return True            # assume there is; the render says so if not


def leading_video_gap(source: Path) -> float:
    """Seconds of audio at the start with no picture behind it.

    --download-sections can only cut video at a keyframe, so a section
    routinely begins with audio while the first video frame arrives a
    second or so later. The file is internally correct -- a given instant
    carries the same timestamp in both streams -- and it still plays wrong
    in the place it matters: a player that honours the gap holds a frozen
    frame, and a player that simply starts decoding, X's among them, runs
    the picture that far behind the sound for the whole clip.

    Reported as audio-start minus video-start, floored at zero. A negative
    result means the video leads, which players handle without help.
    """
    def start(stream: str) -> float | None:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", stream,
                 "-show_entries", "stream=start_time", "-of",
                 "default=nw=1:nk=1", str(source)],
                capture_output=True, text=True, timeout=30).stdout.strip()
            return float(out.splitlines()[0])
        except (ValueError, IndexError, OSError, subprocess.SubprocessError):
            return None

    video, audio = start("v:0"), start("a:0")
    if video is None or audio is None:
        return 0.0
    return max(0.0, video - audio)


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

    # Drop the audio that plays before the first video frame, and start both
    # streams at zero, so every player agrees where the clip begins. Without
    # this the sound leads the picture by whatever the keyframe cost -- 1.3
    # seconds on the clip that went out, for the whole clip.
    #
    # Trimming the audio rather than delaying the video is what keeps the
    # words on the right faces: that leading audio belongs to a moment the
    # section has no picture for, so it is the part with nothing to sync to.
    # A section can come back with no audio at all -- it has, and the render
    # then died on an audio bitrate flag with nothing to apply it to, which
    # reads as a codec error rather than a missing stream.
    has_audio = _has_audio(source)
    steps_audio: list[str] = []
    if has_audio:
        gap = leading_video_gap(source)
        trim = (f"atrim=start={gap:.3f},asetpts=PTS-STARTPTS"
                if gap > 0.04 else "asetpts=PTS-STARTPTS")
        steps_audio = [f"[0:a]{trim}[aud]"]
    reset = "setpts=PTS-STARTPTS,"

    if wide:
        # Cover, not fit: fill the frame and crop the overflow rather than
        # leaving a bar. The source is already 16:9, so this crops nothing
        # in practice and protects the frame if one ever is not.
        steps = [f"[0:v]{reset}scale={size}:{height}:"
                 f"force_original_aspect_ratio=increase,"
                 f"crop={size}:{height}[vid]",
                 "[vid][1:v]overlay=0:0:shortest=1[base]"]
    else:
        steps = [f"[0:v]{reset}scale={size}:-2[vid]",
                 "[1:v][vid]overlay=(W-w)/2:(H-h)/2:shortest=1[base]"]
    label = "base"
    if captions:
        # One input, not one per caption.
        #
        # Every caption used to arrive as its own -i plus its own chained
        # overlay. A 46 second clip is 36 of them and a 180 second clip is
        # 147, and ffmpeg decodes every input in the graph: at 1080x1920 a
        # caption is 8.3 MB of RGBA, so the long clip asked for about 1.2 GB
        # of frame buffers before x264 had allocated anything. It OOMed the
        # web service, which took search down with it, because the render
        # runs in the same process.
        #
        # The concat demuxer plays the same PNGs as a single timed stream,
        # so the graph holds one caption frame at a time however many there
        # are. Memory stops scaling with clip length.
        blank = workdir / "capgap.png"
        _blank_frame(blank, size, height)
        listing = workdir / "captions.txt"
        listing.write_text(_concat_list(captions, workdir, size, height,
                                        blank))
        inputs += ["-f", "concat", "-safe", "0", "-i", str(listing)]
        steps.append(f"[{label}][2:v]overlay=0:0:shortest=1[capped]")
        label = "capped"

    result, peak_mb = run_watching_memory(
        _capped(["ffmpeg", "-y", *inputs,
         "-filter_complex", ";".join(steps + steps_audio),
         "-map", f"[{label}]",
         *(["-map", "[aud]", "-c:a", "aac",
            "-b:a", "256k" if best else "128k"] if has_audio else ["-an"]),
         *_encoder(size, best),
         "-movflags", "+faststart",
         "-shortest", str(out)]),
        timeout=300)
    if peak_mb:
        # The number the ceiling has to be set from. Logged on every render
        # so it is a measurement rather than a guess, and so the figure
        # tracks the canvas rather than being pinned to one that was true
        # in September.
        logger.info("render at %dpx peaked at %.0f MB of address space",
                    size, peak_mb)
    if result.returncode != 0:
        tail = "; ".join((result.stderr or "").strip().splitlines()[-3:])
        # Say which failure this is. Out of memory surfaces as a generic
        # allocation error, and reading that as a broken filter graph is
        # how an instance too small for the canvas gets mistaken for a bug.
        if _looks_like_out_of_memory(result.returncode, tail):
            raise RuntimeError(
                f"render ran out of memory at {size}px — the ceiling is "
                f"{CLIP_MEMORY_LIMIT_MB}MB")
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


def _writable_cookie_jar(source: str | None) -> str | None:
    """A copy of the cookie file that yt-dlp is allowed to write to.

    yt-dlp rewrites the jar after every request, because YouTube rotates
    the session and the refreshed values are what keep it alive. A secret
    file is mounted read-only, so pointing straight at it failed every
    attempt with "[Errno 30] Read-only file system" -- the cookies were
    correct and never got used.

    Copied once per process rather than per fetch, so refreshed cookies
    carry from one clip to the next; a restart takes a clean copy from the
    secret again, which is also the way a corrupted jar repairs itself.
    """
    if not source:
        return None
    origin = Path(source)
    if not origin.is_file():
        logger.warning("YT_COOKIES_FILE is set to %s but there is no file "
                       "there — continuing without cookies", source)
        return None
    try:
        target = Path(tempfile.gettempdir()) / "yt-cookies.txt"
        shutil.copyfile(origin, target)
        target.chmod(0o600)
        logger.info("cookies copied to %s so yt-dlp can refresh them",
                    target)
        return str(target)
    except OSError as exc:
        logger.warning("could not copy the cookie file (%s) — continuing "
                       "without cookies", exc)
        return None


class ClipService:
    """Background clip jobs, one at a time, with everything bounded."""

    def __init__(self, proxy: str | None = None,
                 cookies: str | None = None):
        self._jobs: dict[str, Job] = {}
        self._gate = asyncio.Semaphore(MAX_CONCURRENT)
        self._proxy = proxy
        self._cookies = _writable_cookie_jar(cookies)
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
                # 200 was enough for one message and is not enough for a
                # trail of three. The last attempt's line was being cut off
                # entirely, which hid the one that mattered.
                job.error = str(exc)[:600]
                logger.warning("clip %s failed: %s", job.id, job.error)

    def _build(self, episode: dict, start: float,
               end: float, out: Path) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            raw = work / "raw.mp4"
            # Same recipe make_clip.py uses for a clip that goes out in
            # public: full height, 16:9, quality-targeted encode. The old
            # defaults here were 720 square at a fixed bitrate, which is a
            # preview — and what a viewer shares is the clip itself, so it
            # should be the quality a clip going out in public wants.
            fetch_section(episode["url"], start, end, raw, self._proxy,
                          height=CLIP_HEIGHT, cookies=self._cookies)
            backdrop = work / "backdrop.png"
            make_wide_overlay(episode.get("title", ""), stamp(start),
                              backdrop, CLIP_HEIGHT,
                              int(CLIP_HEIGHT * 9 / 16))
            captions = build_captions(episode.get("segments", []), start, end)
            render(raw, captions, backdrop, work, out, CLIP_HEIGHT,
                   best=True, wide=True)
