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
from urllib.parse import urlsplit, urlunsplit

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
        cmd += ["--remote-components", "ejs:github"]
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
    clients = PLAYER_CLIENTS if is_youtube else ("",)
    attempts = max(3, len(clients))
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
        if proxy:
            pinned = pin_one_exit_ip(proxy, uuid.uuid4().hex[:12])
            run = [*run[:-3], "--proxy", pinned, *run[-3:]]
            env = {**os.environ,
                   "http_proxy": pinned, "https_proxy": pinned,
                   "HTTP_PROXY": pinned, "HTTPS_PROXY": pinned}
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
                    logger.info("fetched on attempt %d of %d (client %s)",
                                attempt, attempts, client or "default")
                return
            # Exit code 0 and a playable file, with a hole in it.
            last = (f"the download is missing {gap:.1f}s of video "
                    f"(a dropped fragment)")
            logger.warning("attempt %d came back with a %.1fs gap — "
                           "discarding it", attempt, gap)
        else:
            last = ((result.stderr or "").strip().splitlines()
                    or ["(no stderr)"])[-1]
        # A partial file from the failed attempt would make the next one
        # look like it succeeded.
        dest.unlink(missing_ok=True)
        if attempt < attempts:
            logger.info("fetch attempt %d failed on client %s (%s) — "
                        "retrying on a new exit IP",
                        attempt, client or "default", last[:90])
    raise RuntimeError(f"could not fetch that section: {last[:180]}")


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
